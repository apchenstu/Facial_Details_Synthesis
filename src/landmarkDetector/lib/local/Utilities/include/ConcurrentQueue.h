//
// Copyright (c) 2013 Juan Palacios juan.palacios.puyana@gmail.com
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef CONCURRENT_QUEUE_
#define CONCURRENT_QUEUE_

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

template <typename T>
class ConcurrentQueue
{
public:

	T pop()
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_empty_.wait(mlock);
		}
		auto val = queue_.front();
		queue_.pop();
		mlock.unlock();
		cond_full_.notify_one();
		return val;
	}

	void pop(T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_empty_.wait(mlock);
		}
		item = queue_.front();
		queue_.pop();
		mlock.unlock();
		cond_full_.notify_one();
	}

	void push(const T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);

		while (capacity_ > 0 && queue_.size() >= capacity_)
		{
			cond_full_.wait(mlock);
		}
		queue_.push(item);
		mlock.unlock();
		cond_empty_.notify_one();
	}
	
	void set_capacity(int capacity)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		capacity_ = capacity;
	}

	bool empty()
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		return queue_.empty();
	}

	ConcurrentQueue() = default;
	ConcurrentQueue(const ConcurrentQueue&) = delete;            // disable copying
	ConcurrentQueue& operator=(const ConcurrentQueue&) = delete; // disable assignment

private:
	std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_empty_;
	std::condition_variable cond_full_;
	// If capacity greater than one, the queue will block on push if there are too many elements in it
	int capacity_ = 0;
};

#endif