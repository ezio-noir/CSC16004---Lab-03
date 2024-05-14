#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

// Thread-pooling manager
class ThreadPool {
private:
	vector<thread> _threads;
	queue<function<void()>> _tasks;
	mutex _mutex;
	condition_variable _cv;
	bool _stop = false;

private:
	// Thread loop
	void workerThread() {
		// When thread is idle, check task queue for available task
		while (true) {
			function<void()> task;
			{
				unique_lock<mutex> lock(_mutex);
				_cv.wait(lock, [this] { return !_tasks.empty() || _stop; });
				if (_stop && _tasks.empty()) return;
				task = move(_tasks.front());
				_tasks.pop();
			}
			task();
		}
	}

public:
	// Initialize thread pool with gien number of threads, if <= 0, then get the maxium
	// threads available in CPU
	ThreadPool(int numThreads) {
		numThreads = (numThreads <= 0) ? thread::hardware_concurrency() : numThreads;
		_threads = vector<thread>();
		for (int i = 0; i < numThreads; ++i) {
			_threads.emplace_back(bind(&ThreadPool::workerThread, this));
		}
	}

	~ThreadPool() {
		{
			unique_lock<mutex> lock(_mutex);
			_stop = true;
		}
		_cv.notify_all();
		for (thread& thread : _threads) {
			thread.join();
		}
	}

public:
	// Enqueue function as task
	void enqueue(function<void(int)> func, int otv) {
		{
			unique_lock<mutex> lock(_mutex);
			_tasks.emplace([=] { func(otv); });
		}
		_cv.notify_one();
	}

	// Enqueue function as task
	void enqueue(function<void(int, int)> func, int otv, int layer) {
		{
			unique_lock<mutex> lock(_mutex);
			_tasks.emplace([=] { func(otv, layer); });
		}
		_cv.notify_one();
	}

	// Check if queue is empty
	bool isEmpty() {
		return _tasks.empty();
	}
};
