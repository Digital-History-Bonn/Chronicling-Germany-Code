"""Module for Predictor class that handles multiprocessing and file management."""

import json
import os
import traceback
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm


def run_process(
    predict_function: Callable,
    init_model_function: Callable,
    queue: Queue,
    failed_queue: Queue,
    done_queue: Queue,
    num_threads: int,
    model_args: list,
    page_level_threads: bool,
    save_done: bool,
) -> None:
    """
    Takes paths from a multiprocessing queue and must be terminated externally when all paths have been processed.
    Args:
        queue: multiprocessing queue for path tuples.
        failed_queue:  multiprocessing queue for image paths, where the prediction has failed.
        done_queue:  multiprocessing queue for image paths, where the image has been processed completely.
        thread_count: this number of threads will run predictions in parallel. This might lead to an CUDA out of
        memory error if too many threads are launched.
        page_level_threads: activates threads that are launched at this level instead of inside the
        predict function.
    """
    assert num_threads > 0

    model = init_model_function(*model_args)
    threads = []

    if page_level_threads:
        launch_threads(
            done_queue,
            failed_queue,
            model,
            num_threads,
            predict_function,
            queue,
            save_done,
        )
    else:
        while True:
            args = queue.get()
            if args[-1]:
                break
            try:
                thread: Union[None, Thread] = predict_function(args, model)
                if thread is not None:
                    threads.append(thread)
                if len(threads) >= num_threads:
                    join_threads(threads)
                    threads = []
                if save_done:
                    done_queue.put(args[0], block=True)
            except Exception as e:  # pylint: disable=broad-exception-caught
                failed_queue.put(args[0], block=True)
                print(e)
                traceback.print_exc()


def launch_threads(
    done_queue: Queue,
    failed_queue: Queue,
    model: object,
    num_threads: int,
    predict_function: Callable,
    queue: Queue,
    save_done: bool,
) -> None:
    """
    Launch threads for prediction and join them after completion.
    Args:
        failed_queue:  multiprocessing queue for image paths, where the prediction has failed.
        done_queue:  multiprocessing queue for image paths, where the image has been processed completely.
    """
    while True:
        threads: List[Thread] = []
        for i in range(num_threads):
            args = queue.get()
            if args[-1]:
                join_threads(threads)
                return

            threads.append(
                Thread(
                    target=run_thread,
                    args=(
                        args,
                        predict_function,
                        failed_queue,
                        done_queue,
                        model,
                        save_done,
                    ),
                )
            )
            threads[i].start()
        join_threads(threads)


def run_thread(
    args: list,
    predict_function: Callable,
    failed_queue: Queue,
    done_queue: Queue,
    model: object,
    save_done: bool,
) -> None:
    """
    Takes paths from a multiprocessing queue, that are processed with the same model in different threads.#
    Args:
        failed_queue:  multiprocessing queue for image paths, where the prediction has failed.
        done_queue:  multiprocessing queue for image paths, where the image has been processed completely.
    """
    try:
        predict_function(args, model)
        if save_done:
            done_queue.put(args[0], block=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
        failed_queue.put(args[0], block=True)
        print(e)
        traceback.print_exc()


def join_threads(threads: List[Thread]) -> None:
    """
    Join all threads.
    """
    for thread in threads:
        thread.join()


# make super class that is not intended for prediction?
class MPPredictor:
    """Class for handling multiprocessing for prediction and can be used with an arbitrary model."""

    def __init__(
        self,
        name: str,
        predict_function: Callable,
        init_model_function: Callable,
        path_queue: Queue,
        model_list: list,
        data_path: str,
        save_done: bool = False,
        page_level_threads: bool = False,
    ) -> None:
        self.name = name
        self.predict_function = predict_function
        self.path_queue = path_queue
        self.model_list = model_list
        self.data_path = Path(
            data_path
        )  # todo: change this to expect path instead of string
        self.init_model_function = init_model_function
        self.save_done = save_done
        self.page_level_threads = page_level_threads

    def launch_processes(
        self,
        num_gpus: int = 0,
        num_threads: int = 1,
        total: Optional[int] = None,
        get_progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Launches processes and handles multiprocessing Queues.
        """
        if not total:
            total = self.path_queue.qsize()
        if not get_progress:
            get_progress = {
                "method": get_queue_progress,
                "args": (total, self.path_queue),
            }
        if num_gpus > 0:
            print(f"Using {num_gpus} gpu device(s).")
        else:
            print("Using cpu.")

        failed_queue: Queue = Queue()
        done_queue: Queue = Queue()

        # todo: fill path queue after process start to avoid a full queue exception.
        # todo: add proper debug option

        # DEBUG RUN WITHOUT PROCESS
        # run_process(
        #             self.predict_function,
        #             self.init_model_function,
        #             self.path_queue,
        #             failed_queue,
        #             done_queue,
        #             num_threads,
        #             self.model_list[0],
        #             self.page_level_threads,
        #             self.save_done,
        #         )
        # sleep(1)
        # self.path_queue.put(("", "", "", True))

        processes = [
            Process(
                target=run_process,
                args=(
                    self.predict_function,
                    self.init_model_function,
                    self.path_queue,
                    failed_queue,
                    done_queue,
                    num_threads,
                    self.model_list[i if num_gpus > 0 else 0],
                    self.page_level_threads,
                    self.save_done,
                ),
            )
            for i in range(len(self.model_list))
        ]
        for process in processes:
            process.start()

        done_file, done_list, failed_dict, failed_file = self.init_logs()

        # todo: merge this with run processes?
        with tqdm(total=total, desc=self.name, unit="pages") as pbar:
            while True:
                self.empty_log_queues(done_list, done_queue, failed_dict, failed_queue)
                progress = get_progress["method"](*get_progress["args"])
                pbar.n = progress
                pbar.refresh()
                sleep(1)
                if progress >= total:
                    break
        for _ in processes:
            self.path_queue.put(("", "", "", True))
        self.empty_log_queues(done_list, done_queue, failed_dict, failed_queue)
        for process in tqdm(processes, desc="Waiting for processes to end"):
            process.join()

        self.save_logs(done_file, done_list, failed_dict, failed_file)

    def save_logs(
        self,
        done_file: Path,
        done_list: List[str],
        failed_dict: Dict[str, List[str]],
        failed_file: Path,
    ) -> None:
        """
        Save logs for failed images and images that have been completely processed.
        """
        with open(failed_file, "w", encoding="utf-8") as file:
            json.dump(failed_dict, file)
        with open(done_file, "w", encoding="utf-8") as file:
            json.dump(done_list, file)

    def empty_log_queues(
        self,
        done_list: List[str],
        done_queue: Queue,
        failed_dict: Dict[str, List[str]],
        failed_queue: Queue,
    ) -> None:
        """
        Empty log queues during execution to avoid an overflow of a queue.
        """
        while not failed_queue.empty():
            failed_dict[self.name].append(failed_queue.get())
        while not done_queue.empty():
            done_list.append(done_queue.get())

    def init_logs(self) -> Tuple[Path, List[str], Dict[str, List[str]], Path]:
        """
        Initialize done and failed logs. If files are already present, they are loaded and appended to.
        Otherwise, logs are created empty.
        """
        log_path = self.data_path / "logs"
        failed_file = log_path / "failed.json"
        done_file = log_path / "done.json"
        if not log_path.is_dir():
            os.makedirs(log_path)
        failed_dict = {}
        if failed_file.is_file():
            with open(failed_file, encoding="utf-8") as file:
                failed_dict = json.load(file)
        done_list = []
        if done_file.is_file():
            with open(done_file, encoding="utf-8") as file:
                done_list = json.load(file)
        if self.name not in failed_dict.keys():
            failed_dict[self.name] = []
        return done_file, done_list, failed_dict, failed_file


def run_processes(
    get_progress: Dict[str, Any],
    processes: List[Process],
    path_queue: Queue,
    total: int,
    name: str,
) -> None:
    """
    Launches basic processes.
    Args:
        get_progress(dict): Dictionary containing the get progress method and its arguments with keys 'method'
        and 'args'.
    """
    # todo: integrate this in MPPredictor
    for process in processes:
        process.start()

    with tqdm(total=total, desc=name, unit="pages") as pbar:
        while True:
            progress = get_progress["method"](*get_progress["args"])
            pbar.n = progress
            pbar.refresh()
            sleep(1)
            if progress >= total:
                break
    for _ in processes:
        path_queue.put(("", True))
    for process in tqdm(processes, desc="Waiting for processes to end"):
        process.join()


def get_cpu_count() -> int:
    """Returns the number of CPUs available."""
    return cpu_count()


def get_queue_progress(total: int, queue: Queue) -> int:
    """Returns difference between total and queue size."""
    return total - queue.qsize()
