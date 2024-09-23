package com.kotlin.interview

import org.junit.Test

import org.junit.Assert.*
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class ExampleUnitTest {
    //@Test
    fun maxArea() {
        println("(mind)Container With Most Water")
        val nums = intArrayOf(1, 3, 4, 6, 2, 9, 8, 5)
        println("output:${maxArea(nums)}")
    }

    private fun maxArea(nums: IntArray): Int {
        var maxArea = 0
        var i = 0
        var r = nums.size - 1
        while (i < r) {
            val height = min(nums[i], nums[r])
            val widget = r - i
            val area = height * widget
            maxArea = max(area, maxArea)
            if (nums[i] < nums[r]) {
                i++
            } else {
                r--
            }
        }
        println("Time complexity O(n)")
        println("Space complexity O(1)")
        return maxArea
    }

    //@Test
    fun validMountainArray() {
        println("(easy)Valid Mountain Array")
        val arr = intArrayOf(1, 3, 4, 6, 4, 3, 2, 1)
        println("result:${validMountainArray(arr)}")
    }

    private fun validMountainArray(arr: IntArray): Boolean {
        var i = 1
        //check for increasing
        while (i < arr.size && arr[i] > arr[i - 1]) {
            i++
        }
        return if (i == 1 || i == arr.size) {
            false
        } else {
            //check for decreasing
            while (i < arr.size && arr[i] < arr[i - 1]) {
                i++
            }
            i == arr.size
        }
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }

    fun numRescueBoats() {
        println("(mind)Boats to save people")
        val people = intArrayOf(1, 2, 1, 3, 2, 3, 1, 2, 3, 2, 1)
        val limit = 3
        println("output:${numRescueBoats(people, limit)}")
        println("time complexity O(nlogn)")
        println("space complexity O(n)")
    }

    private fun numRescueBoats(people: IntArray, limit: Int): Int {
        var min = 0
        var max = people.size - 1
        var boats = 0
        people.sort()
        while (min <= max) {
            if (people[min] + people[max] <= limit) {
                min++
            }
            max--
            boats++
        }
        return boats
    }

    fun moveZeroes() {
        println("(easy)Movie zeros")
        val nums = intArrayOf(0, 1, 0, 3, 12, 9, 0)
        moveZeroes(nums)
        println("output:${nums.joinToString()}")
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }

    private fun moveZeroes(nums: IntArray) {
        var offset = 0
        for (i in nums.indices) {
            if (nums[i] == 0) {
                offset++
            } else {
                nums[i - offset] = nums[i]
                if (offset > 0) {
                    nums[i] = 0
                }
            }
        }
    }

    //@Test
    fun lengthOfLongestSubstring() {
        println("(mind)Longest Substring Without Repeating Characters")
        val s = "abcdefbdabdcaadbeb"
        println("output:${lengthOfLongestSubstring(s)}")
    }

    private fun lengthOfLongestSubstring(s: String): Int {
        var maxLength = 0
        val charSet = mutableSetOf<Char>()
        var start = 0
        var end = 0
        while (end < s.length) {
            if (!charSet.contains(s[end])) {
                charSet.add(s[end])
                end++
                maxLength = max(maxLength, charSet.size)
            } else {
                charSet.remove(s[start])
                start++
            }
        }
        return maxLength
        println("Time complexity O(n)")
        println("Space complexity O(min(m, n)")
    }

    //@Test
    fun searchRange() {
        println("(easy)Find first and last position of element in sorted array")
        val nums = intArrayOf(5, 7, 7, 8, 8, 10)
        val target = 8
        println("output:${searchRange(nums, target).joinToString()}")
    }

    private fun searchRange(nums: IntArray, target: Int): IntArray {
        if (nums.isEmpty() || target == -1) return intArrayOf(-1, -1)
        return intArrayOf(findFirstPosition(nums, target), findLastPosition(nums, target))
        println("Time complexity O(logn)")
        println("Space complexity O(1)")
    }

    private fun findFirstPosition(input: IntArray, target: Int): Int {
        var start = 0
        var end = input.size - 1
        var result = -1
        while (start <= end) {
            val mid = start + (end - start) / 2
            if (input[mid] < target) {
                start = mid + 1
            } else {
                end = mid - 1
                if (input[mid] == target) {
                    result = mid
                }
            }
        }
        return result
    }

    private fun findLastPosition(input: IntArray, target: Int): Int {
        var start = 0
        var end = input.size - 1
        var result = -1
        while (start <= end) {
            val mid = start + (end - start) / 2
            if (input[mid] > target) {
                end = mid - 1
            } else {
                start = mid + 1
                if (input[mid] == target) {
                    result = mid
                }
            }
        }
        return result
    }

    fun firstBadVersion() {
        println("(easy)first bad version")
        val n = 10
        println("output:${firstBadVersion(n)}")
    }

    private fun isBadVersion(version: Int): Boolean {
        return version >= 4
        println("Time complexity O(logn)")
        println("Space complexity O(1)")
    }

    private fun firstBadVersion(n: Int): Int {
        if (n == 1) return 1
        var start = 1
        var end = n
        while (start < end) {
            val mid = start + (end - start) / 2
            if (isBadVersion(mid)) {
                end = mid
            } else {
                start = mid + 1
            }
        }
        return start
    }

    fun missingNumber() {
        println("(easy)missing number")
        val nums = intArrayOf(3, 0, 1)
        println("output:${missingNumber(nums)}")
    }

    private fun missingNumber(nums: IntArray): Int {
        val size = nums.size
        val sum = size * (size + 1) / 2
        return sum - nums.sum()
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }

    @Test
    fun countPrimes() {
        println("(mind)count primes")
        val n = 100
        println("result:${countPrimes(n)}")
    }

    private fun countPrimes(n: Int): Int {
        if (n < 2) return 0
        val isPrime = BooleanArray(n) { true }
        isPrime[0] = false
        isPrime[1] = false
        var count = 0
        for (i in 2 until n) {
            if (isPrime[i]) {
                count++
                var j = i.toLong() * i // Use Long to avoid overflow
                while (j < n) {
                    isPrime[j.toInt()] = false
                    j += i
                }
            }
        }
        return count
        println("Time complexity O(nloglogn)")
        println("Space complexity O(n)")
    }

    fun singleNumber() {
        println("(easy)single number")
        val nums = intArrayOf(4, 1, 2, 1, 2)
        println("result:${singleNumber(nums)}")
    }

    private fun singleNumber(nums: IntArray): Int {
        var single = 0
        for (num in nums) {
            single = single xor num
        }
        return single
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }


}