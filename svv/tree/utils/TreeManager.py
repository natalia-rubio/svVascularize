import threading
from threading import Lock, Condition
from scipy.spatial import cKDTree
import numpy as np
from time import perf_counter
from threading import Lock, Condition
import os
from usearch.index import Index, MetricKind
MAX_THREADS = min(4, max(1, os.cpu_count() - 1))

class USearchTree:
    def __init__(self, data, space='l2sq', ef_construction=200, M=100, ef=20):
        """
        Initialize USearch tree with the given data.

        Parameters:
        - data: A 2D numpy array of shape (N, D), where N is the number of points, and D is the dimensionality.
        - space: Metric to use for distance calculation. Options: 'l2' (Euclidean), 'ip' (inner product), 'cos' (cosine similarity).
        - ef_construction: The size of the dynamic candidate list during construction.
        - M: Controls the number of bi-directional links created for every new element during construction.
        """
        self.dim = data.shape[1]
        metric = {'l2': 'euclidean', 'ip': 'dot', 'cos': 'cos'}.get(space, 'euclidean')

        self.index = Index(
            ndim=self.dim,
            metric=space,
            connectivity=M,
            expansion_add=ef_construction,
            expansion_search=ef,
            dtype='f32'
        )

        labels = np.arange(data.shape[0])
        self.index.add(labels, data)

    def query(self, points, k=1, threads=MAX_THREADS):
        """
        Query the nearest neighbors for the given points.

        Parameters:
        - points: A 2D numpy array of shape (N, D) where N is the number of query points.
        - k: The number of nearest neighbors to return.

        Returns:
        - distances: A 2D array of distances to the nearest neighbors.
        - indices: A 2D array of indices of the nearest neighbors.
        """
        results = self.index.search(points, k, threads=threads)
        distances = results.distances
        indices = results.keys
        return distances, indices

    def query_ball_point(self, points, r, **kwargs):
        """
        Query points within distance `r` from the given points.

        Parameters:
        - points: A 2D numpy array of shape (N, D) where N is the number of query points.
        - r: The radius to search within.

        Returns:
        - A list of lists, where each sublist contains the indices of points within distance `r` of the corresponding query point.
        """
        # Approximate a ball query by performing knn search with a large `k` and filtering by `r`
        k = kwargs.get("k", min(50, self.index.size))
        distances, indices = self.query(points, k)

        # Filter based on radius `r`
        result = []
        for i in range(len(points)):
            within_r = indices[i][distances[i] <= r]  # Filter neighbors by radius
            result.append(within_r)

        return result

    def set_ef(self, ef):
        """
        Set the `ef` parameter, which controls the search depth and the trade-off between speed and accuracy.

        Parameters:
        - ef: The size of the dynamic list for nearest neighbors search.
        """
        self.index.expansion_search = ef

    def add_items(self, data, ids):
        """
        Add more points to the USearch index.

        Parameters:
        - data: A 2D numpy array of new points to add.
        """
        self.index.add(ids, data)

    def replace(self, data, ids):
        """
        Replace points in the index.

        Parameters:
        - data: A 2D numpy array of new points to replace existing points.
        - ids: A list of IDs of the points to replace.
        """
        self.index.remove(ids)
        self.index.add(ids, data)

class ReadWriteLock:
    def __init__(self):
        self._read_ready = Condition(Lock())
        self._readers = 0

    def acquire_read(self):
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()


class KDTreeManager:
    def __init__(self, data):
        self.active_tree = cKDTree(data)
        self.previous_tree = None
        self.update_tree = None
        self.tree_lock = threading.Lock()
        self.update_done_event = threading.Event()
        self.update_done_event.set()
        self.update_event = threading.Event()
        self.shutdown_event = threading.Event()

        # This variable will hold the latest update request
        self.update_queue = []
        self.update_queue_lock = threading.Lock()

        # Start a background thread that will handle updates
        self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
        self.update_thread.start()

    def start_update(self, new_data):
        with self.update_queue_lock:
            # Store the new data as the latest pending update
            self.update_queue.append(new_data)

        # If there's no current update thread or it's finished, start a new one
        #if self.update_thread is None or not self.update_thread.is_alive():
        #    self._start_new_update()
        self.update_done_event.clear()
        self.update_event.set()

    def _update_worker(self):
        while not self.shutdown_event.is_set():
            # Wait for an update event
            self.update_event.wait()

            # Process all updates in the queue
            while True:
                with self.update_queue_lock:
                    if not self.update_queue:
                        break
                    new_data = self.update_queue.pop(-1)
                    self.update_queue = []

                # Build the new KDTree
                new_tree = cKDTree(new_data)
                # Swap the trees safely
                with self.tree_lock:
                    self.update_tree = new_tree

                # Immediately swap active trees if no further updates are queued
                #if not self.update_queue:
                #    self.swap_trees()
                self.update_done_event.set()
            # Clear the event after processing all updates
            self.update_event.clear()

    #def _start_new_update(self):
    #    with self.update_queue_lock:
    #        if self.pending_update is None:
    #            return  # No update to start
    #
    #        # Retrieve and clear the pending update
    #        new_data = self.pending_update
    #        self.pending_update = None
    #
    #    # Start a new update thread
    #    self.update_done_event.clear()
    #    self.update_thread = threading.Thread(target=self._update_tree, args=(new_data,))
    #    self.update_thread.start()

    #def _update_tree(self, new_data):
    #    # Build the new KDTree
    #    new_tree = cKDTree(new_data)
    #    # Swap the trees safely
    #    with self.tree_lock:
    #        self.update_tree = new_tree
    #
    #    # Signal that the update is done
    #    self.update_done_event.set()
    #
    #    # Check if there is another update waiting to be processed
    #    self._start_new_update()

    def swap_trees(self):
        with self.tree_lock:
            if self.update_tree is None:
                return
            self.previous_tree = self.active_tree
            self.active_tree, self.update_tree = self.update_tree, self.active_tree
            self.update_tree = None

    def query(self, points, k=1):
        #with self.tree_lock:
        result = self.active_tree.query(points, k=k)
        return result

    def query_ball_point(self, points, r):
        #with self.tree_lock:
        result = self.active_tree.query_ball_point(points, r)
        return result

    def query_ball_tree(self, other, r, eps=0.0):
        with self.tree_lock:
            result = self.active_tree.query_ball_tree(other, r, eps=eps)
            return result

    def revert_to_previous(self):
        with self.tree_lock:
            if self.previous_tree is not None:
                self.active_tree, self.previous_tree = self.previous_tree, self.active_tree
                print("Reverted to the previous KDTree.")
            else:
                print("No previous KDTree to revert to.")

    def wait_for_update(self):
        while True:
            # Wait for the current update to complete
            self.update_done_event.wait()

            # Check if there's another pending update
            with self.update_queue_lock:
                if not self.update_queue:
                    break  # No more updates, exit loop
                else:
                    self.update_done_event.clear()
                    self.update_event.set()
                    self._update_worker()

    def shutdown(self):
        # Signal the update thread to shut down
        self.shutdown_event.set()
        self.update_event.set()  # Wake up the thread to exit

        # Wait for the thread to finish
        self.update_thread.join()
