"""Module for Predictor class that handles multiprocessing and file management."""
import json
import os
from multiprocessing import Queue, Process
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Callable, List

from tqdm import tqdm


# TODO: create an abstract class(interface) for prediction that is implemented in layout, baseline and ocr versions
#  and used here insead of multiple functions. Create Class which contains and handles all Queues, such as bools.
def run_process(predict_function: Callable, init_model_function: Callable, queue: Queue, failed_queue: Queue,
                done_queue: Queue, num_threads: int, model_args: list, page_level_threads: bool,
                save_done: bool) -> None:
    """
    Takes paths from a multiprocessing queue and must be terminated externally when all paths have been processed.
    Args:
        queue: multiprocessing queue for path tuples.
        failed_queue:  multiprocessing queue for image paths, where the prediction has failed.
        done_queue:  multiprocessing queue for image paths, where the image has been processed completely.
        thread_count: this number of threads will run predictions in parallel. This might lead to an CUDA out of
        memory error if too many threads are launched.
        :param page_level_threads: activates threads that are launched at this level instead of inside the
        predict function.
    """
    model = init_model_function(*model_args)
    while True:
        if page_level_threads:
            launch_threads(done_queue, failed_queue, model, num_threads, predict_function, queue, save_done)
        else:
            args = queue.get()
            if args[-1]:
                break
            try:
                predict_function(args, model)
            except Exception as e:
                failed_queue.put(args[0])
                print(e)
            if save_done:
                done_queue.put(args[0])


def launch_threads(done_queue: Queue, failed_queue: Queue, model: object, num_threads: int, predict_function: Callable,
                   queue: Queue, save_done: bool) -> None:
    """
    Launch threads for prediction and join them after completion.
    Args:
        failed_queue:  multiprocessing queue for image paths, where the prediction has failed.
        done_queue:  multiprocessing queue for image paths, where the image has been processed completely.
    """
    threads: List[Thread] = []
    for i in range(num_threads):
        args = queue.get()
        if args[-1]:
            break

        threads.append(Thread(target=run_thread, args=(args, predict_function, failed_queue,
                                                       done_queue, model, save_done)))
        threads[i].start()
    join_threads(threads)


def run_thread(args: list, predict_function: Callable, failed_queue: Queue,
               done_queue: Queue, model: object, save_done: bool) -> None:
    """
    Takes paths from a multiprocessing queue, that are processed with the same model in different threads.#
    Args:
        failed_queue:  multiprocessing queue for image paths, where the prediction has failed.
        done_queue:  multiprocessing queue for image paths, where the image has been processed completely.
    """
    try:
        predict_function(args, model)
    except Exception as e:
        failed_queue.put(args[0])
        print(e)
    if save_done:
        done_queue.put(args[0])


def join_threads(threads: List[Thread]) -> None:
    """
    Join all threads.
    """
    for thread in threads:
        thread.join()


class MPPredictor:
    """Class for handling multiprocessing for prediction and can be used with an arbitrary model."""

    def __init__(self, name: str, predict_function: Callable, init_model_function: Callable, path_queue: Queue,
                 model_list: list, data_path: str, save_done: bool, page_level_threads) -> None:
        self.name = name
        self.predict_function = predict_function
        self.path_queue = path_queue
        self.model_list = model_list
        self.data_path = Path(data_path)
        self.init_model_function = init_model_function
        self.save_done = save_done
        self.page_level_threads = page_level_threads

    def launch_processes(self, num_gpus: int = 0, num_threads: int = 1) -> None:
        """
        Launches processes and handles multiprocessing Queues.
        """
        if num_gpus > 0:
            print(f"Using {num_gpus} gpu device(s).")
        else:
            print("Using cpu.")

        failed_queue: Queue = Queue()
        done_queue: Queue = Queue()
        total = self.path_queue.qsize()

        processes = [Process(target=run_process,
                             args=(
                                 self.predict_function, self.init_model_function, self.path_queue, failed_queue,
                                 done_queue, num_threads, self.model_list[i if num_gpus > 0 else 0],
                                 self.page_level_threads, self.save_done
                             ))
                     for i in range(len(self.model_list))]
        for process in processes:
            process.start()

        with tqdm(total=total, desc=self.name, unit="pages") as pbar:
            while not self.path_queue.empty():
                pbar.n = total - self.path_queue.qsize()
                pbar.refresh()
                sleep(1)
        for _ in processes:
            self.path_queue.put(("", "", "", True))
        for process in tqdm(processes, desc="Waiting for processes to end"):
            process.join()

        self.save_logs(done_queue, failed_queue)

    def save_logs(self, done_queue: Queue, failed_queue: Queue) -> None:
        """
        Save logs for failed images and images that have been completely processed.
        Args:
            failed_queue:  multiprocessing queue for image paths, where the prediction has failed.
            done_queue:  multiprocessing queue for image paths, where the image has been processed completely.
        """
        if not os.path.exists(self.data_path / 'logs'):
            os.makedirs(self.data_path / 'logs')
        if not os.path.exists(self.data_path / 'logs' / 'failed.json'):
            failed_dict = {}
        else:
            with open(self.data_path / 'logs' / 'failed.json', encoding="utf-8") as file:
                failed_dict = json.load(file)
        if not os.path.exists(self.data_path / 'logs' / 'done.json'):
            done_list = []
        else:
            with open(self.data_path / 'logs' / 'done.json', encoding="utf-8") as file:
                done_list = json.load(file)
        if self.name not in failed_dict.keys():
            failed_dict[self.name] = []
        while not failed_queue.empty():
            failed_dict[self.name].append(failed_queue.get())
        with open(self.data_path / 'logs' / 'failed.json', encoding="utf-8") as file:
            json.dump(failed_dict, file)
        while not done_queue.empty():
            done_list.append(done_queue.get())
        with open(self.data_path / 'logs' / 'done.json', encoding="utf-8") as file:
            json.dump(done_list, file)
