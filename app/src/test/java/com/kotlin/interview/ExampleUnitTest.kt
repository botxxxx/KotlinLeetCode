package com.kotlin.interview

import org.junit.Test
import java.util.LinkedList
import java.util.Stack

import kotlin.math.max
import kotlin.math.min

class ExampleUnitTest {

    @Test //02
    fun maxArea() {
        println("(mind)Container With Most Water")
        val nums = intArrayOf(1, 3, 4, 6, 2, 9, 8, 5)
        println("output:${maxArea(nums)}")
        println("Time complexity O(n)")
        println("Space complexity O(1)")
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
        return maxArea
    }

    @Test //03
    fun validMountainArray() {
        println("(easy)Valid Mountain Array")
        val arr = intArrayOf(1, 3, 4, 6, 4, 3, 2, 1)
        println("result:${validMountainArray(arr)}")
        println("Time complexity O(n)")
        println("Space complexity O(1)")
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
    }

    @Test //04
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

    @Test //05
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

    @Test //06
    fun lengthOfLongestSubstring() {
        println("(mind)Longest Substring Without Repeating Characters")
        val s = "abcdefbdabdcaadbeb"
        println("output:${lengthOfLongestSubstring(s)}")
        println("Time complexity O(n)")
        println("Space complexity O(min(m, n)")
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
    }

    @Test //07
    fun searchRange() {
        println("(easy)Find first and last position of element in sorted array")
        val nums = intArrayOf(5, 7, 7, 8, 8, 10)
        val target = 8
        println("output:${searchRange(nums, target).joinToString()}")
        println("Time complexity O(logn)")
        println("Space complexity O(1)")
    }

    private fun searchRange(nums: IntArray, target: Int): IntArray {
        if (nums.isEmpty() || target == -1) return intArrayOf(-1, -1)
        return intArrayOf(findFirstPosition(nums, target), findLastPosition(nums, target))
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

    @Test //08
    fun firstBadVersion() {
        println("(easy)first bad version")
        val n = 10
        println("output:${firstBadVersion(n)}")
        println("Time complexity O(logn)")
        println("Space complexity O(1)")
    }

    private fun isBadVersion(version: Int): Boolean {
        return version >= 4
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

    @Test //09
    fun missingNumber() {
        println("(easy)missing number")
        val nums = intArrayOf(3, 0, 1)
        println("output:${missingNumber(nums)}")
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }

    private fun missingNumber(nums: IntArray): Int {
        val size = nums.size
        val sum = size * (size + 1) / 2
        return sum - nums.sum()
    }

    @Test //10
    fun countPrimes() {
        println("(mind)count primes")
        val n = 100
        println("result:${countPrimes(n)}")
        println("Time complexity O(nloglogn)")
        println("Space complexity O(n)")
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
    }

    @Test //11
    fun singleNumber() {
        println("(easy)single number")
        val nums = intArrayOf(4, 1, 2, 1, 2)
        println("result:${singleNumber(nums)}")
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }

    private fun singleNumber(nums: IntArray): Int {
        var single = 0
        for (num in nums) {
            single = single xor num
        }
        return single
    }

    @Test //12
    fun robotReturnToOrigin() {
        println("(easy)robot return to origin")
        val moves = "URLDDURL"
        println("result:${judgeCircle(moves)}")
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }

    private fun judgeCircle(moves: String): Boolean {
        var x = 0
        var y = 0
        moves.toCharArray().onEach { c ->
            when (c) {
                'U' -> y++
                'D' -> y--
                'L' -> x--
                'R' -> x++
            }
        }
        return x == 0 && y == 0
    }

    @Test //13
    fun addBinary() {
        println("(easy)add binary")
        val a = "11"
        val b = "1"
        println("result:${addBinary(a, b)}")
        println("Time complexity O(n)")
        println("Space complexity O(n)")
    }

    private fun addBinary(a: String, b: String): String {
        val result = StringBuilder()
        var carry = 0
        var i = a.length - 1
        var j = b.length - 1
        while (i >= 0 || j >= 0 || carry == 1) {
            var sum = carry
            if (i >= 0) {
                sum += a[i].digitToInt()
                i--
            }
            if (j >= 0) {
                sum += b[j].digitToInt()
                j--
            }
            result.insert(0, sum % 2)
            carry = sum / 2
        }
        return result.toString()
    }

    @Test //14
    fun twoSum() {
        println("(easy)two sum")
        val nums = intArrayOf(2, 7, 11, 15)
        val target = 9
        println("result:${twoSum(nums, target).joinToString()}")
        println("Time complexity O(n)")
        println("Space complexity O(n)")
    }

    private fun twoSum(nums: IntArray, target: Int): IntArray {
        if (nums.size == 2) {
            return intArrayOf(0, 1)
        } else {
            val hashMap = HashMap<Int, Int>(nums.size)
            nums.forEachIndexed { index, i ->
                if (hashMap.containsKey(target - i)) {
                    return intArrayOf(hashMap[target - i]!!, index)
                } else {
                    hashMap[i] = index
                }
            }
        }
        return intArrayOf()
    }

    @Test //15
    fun arrayContainsDuplicate() {
        println("(easy)contains duplicate")
        val nums = intArrayOf(1, 2, 3, 1)
        println("result:${containsDuplicate(nums)}")
        println("Time complexity O(n)")
        println("Space complexity O(n)")
    }

    private fun containsDuplicate(nums: IntArray): Boolean {
        val hashSet = HashSet<Int>()
        nums.forEach {
            if (hashSet.contains(it)) {
                return true
            } else {
                hashSet.add(it)
            }
        }
        return false
    }

    @Test //17
    fun majorityElement() {
        println("(easy)majority element")
        val nums = intArrayOf(2, 2, 1, 1, 1, 2, 2)
        println("result:${majorityElement(nums)}")
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }

    private fun majorityElement(nums: IntArray): Int {
        val majorityCount = nums.size / 2
        var majority = 0
        var current = -1
        nums.onEach {
            if (current != it) {
                if (majority == 0) {
                    majority = 1
                    current = it
                } else {
                    majority--
                }
            } else {
                majority++
                if (majority > majorityCount) {
                    return current
                }
            }
        }
        return current
    }

    @Test //18
    fun minimumWindowSubstring() {
        println("(hard)minimum window substring")
        val s = "ADOBECODEBANC"
        val t = "ABC"
        println("result:${minWindow(s, t)}")
        println("Time complexity O(n)")
        println("Space complexity O(n)")
    }

    private fun minWindow(s: String, t: String): String {
        if (s.isEmpty() || t.isEmpty()) return ""

        val charCount = IntArray(128) // Assuming ASCII characters
        for (char in t) {
            charCount[char.code]++
        }

        var left = 0
        var right = 0
        var minWindow = ""
        var minLength = Int.MAX_VALUE
        var count = t.length
        while (right < s.length) {
            val endCode = s[right].code
            if (charCount[endCode] > 0) count--
            charCount[endCode]--
            right++

            while (count == 0) {
                if (right - left < minLength) {
                    minLength = right - left
                    minWindow = s.substring(left, right)
                }
                val startCode = s[left].code
                charCount[startCode]++
                if (charCount[startCode] > 0) count++
                left++
            }
        }
        return minWindow
    }

    @Test //19
    fun groupAnagrams() {
        println("(mind)group anagrams")
        val strs = arrayOf("eat", "tea", "tan", "ate", "nat", "bat")
        println("result:${groupAnagrams(strs)}")
        println("Time complexity O(nklogk)")
        println("Space complexity O(nk)")
    }

    private fun groupAnagrams(strs: Array<String>): List<List<String>> {
        if (strs.size <= 1) return listOf(strs.toList())
        val result = mutableListOf<List<String>>()
        val map = mutableMapOf<String, MutableList<String>>()

        for (str in strs) {
            val charCount = IntArray(26)
            for (char in str) {
                charCount[char - 'a']++
            }

            val key = charCount.contentToString() // Use charCount as key
            map.computeIfAbsent(key) { mutableListOf() }.add(str)
        }

        result.addAll(map.values)
        return result
    }

    @Test //20
    fun LRUCache() {
        println("(mind)LRU cache")
        val lruCache = LRUCache(4)
        lruCache.put(1, 1)
        lruCache.put(2, 2)
        lruCache.get(1)
        lruCache.put(3, 3)
        lruCache.get(2) // Output: -1
        lruCache.put(4, 4)
        lruCache.get(1) // Output: -1
        lruCache.get(3) // Output: 3
        lruCache.get(4) // Output: 4

        println("result:${lruCache.print()}")
        println("Time complexity O(1)")
        println("Space complexity O(n)")
    }

    private class LRUCache(val capacity: Int) {
        val cache = LinkedHashMap<Int, Int>(capacity, 0.75f, true)

        fun print() {
            println("${cache.values.reversed()} lest:" + cache.keys.iterator().next())
        }

        fun get(key: Int): Int {
            val x = cache[key] ?: -1
            print()
            return x
        }

        fun put(key: Int, value: Int) {
            if (!cache.containsKey(key) && cache.size == capacity) {
                cache.remove(cache.keys.iterator().next())
            }
            cache[key] = value
            print()
        }
    }

    @Test //32
    fun maximumDepthOfABinaryTree() {
        println("(easy)maximum depth of binary tree")
        val root = TreeNode(3)
        root.left = TreeNode(9)
        root.right = TreeNode(20)
        root.right!!.left = TreeNode(15)
        root.right!!.right = TreeNode(7)

        val depth = maxDepth(root)
        println("Maximum depth of the tree: $depth")
        println("Time complexity O(n)")
        println("Space complexity O(h)")
    }

    class TreeNode(var `val`: Int) {
        var left: TreeNode? = null
        var right: TreeNode? = null
    }

    private fun arrayToTreeNode(arr: Array<Int?>): TreeNode? {
        if (arr.isEmpty() || arr[0] == null) return null

        val root = TreeNode(arr[0]!!)
        val queue = LinkedList<TreeNode?>()
        queue.add(root)

        var i = 1
        while (i < arr.size) {
            val current = queue.poll()
            if (current != null) {
                if (arr[i] != null) {
                    current.left = TreeNode(arr[i]!!)
                    queue.add(current.left)
                }
                i++
                if (i < arr.size && arr[i] != null) {
                    current.right = TreeNode(arr[i]!!)
                    queue.add(current.right)
                }
                i++
            }
        }

        return root
    }

    private fun maxDepth(root: TreeNode?, currentDepth: Int = 1): Int = when {
        root == null -> 0
        root.left == null && root.right == null -> currentDepth
        else -> {
            val right = maxDepth(root.right, currentDepth + 1)
            val left = maxDepth(root.left, currentDepth + 1)
            max(right, left)
        }
    }

    @Test //35
    fun kthSmallestInBST() {
        println("(mind)kth smallest element in a BST")
        val arr = arrayOf(9, 5, 14, 3, 7, null, null, 1)
        val root = arrayToTreeNode(arr)
        println("result:${arr.indices.filter { it != 0 }.joinToString { "${kthSmallest(root, it)}" }}")
        println("Time complexity O(n)")
        println("Space complexity O(h)")
    }

    private fun kthSmallest(root: TreeNode?, k: Int): Int {
        if (root == null || k <= 0) return -1
        var count = 0
        var result = -1
        fun inorder(node: TreeNode?) {
            if (node?.left == null || count >= k) return
            inorder(node.left)
            count++
            if (count == k) {
                result = node.`val`
                return
            } else {
                inorder(node.right)
            }
        }
        inorder(root)
        return result
    }

    @Test //36
    fun serializeAndDeserializeBinaryTree() {
        println("(hard)serialize and deserialize binary tree")
        val treeNode = TreeNode(1).apply {
            left = TreeNode(2)
            right = TreeNode(3)
            right?.left = TreeNode(4)
            right?.right = TreeNode(5)
        }
        val serialize = serialize(treeNode)
        println("serialize:$serialize")
        val deserialize = deserialize(serialize)
        println("deserialize:${serialize(deserialize)}")
        println("Time complexity O(n)")
        println("Space complexity O(n)")
    }

    private fun serialize(root: TreeNode?): String {
        if (root == null) return ""

        val sb = StringBuilder()
        val queue = LinkedList<TreeNode?>()
        queue.add(root)

        while (queue.isNotEmpty()) {
            val node = queue.poll()
            if (node != null) {
                if (sb.isNotEmpty()) {
                    sb.append(",")
                }
                sb.append(node.`val`)
                queue.add(node.left)
                queue.add(node.right)
            } else {
                if (sb.isNotEmpty()) {
                    sb.append(",")
                }
                sb.append("null")
            }
        }
        return sb.toString()
    }

    private fun deserialize(data: String): TreeNode? {
        if (data.isEmpty()) return null
        val arr = data.split(",").toTypedArray()
        val root = TreeNode(arr[0].toInt())
        var i = 1
        val queue = LinkedList<TreeNode?>()
        queue.add(root)
        while (i < arr.size) {
            val node = queue.poll()
            if (node != null) {
                if (arr[i] != "null") {
                    node.left = TreeNode(arr[i].toInt())
                    queue.add(node.left)
                }
                i++
                if (i < arr.size && arr[i] != "null") {
                    node.right = TreeNode(arr[i].toInt())
                    queue.add(node.right)
                }
                i++
            }
        }
        return root
    }

    @Test //37
    fun binaryTreeMaximumPathSum() {
        println("(hard)binary tree maximum path sum")
        val arr: Array<Int?> = arrayOf(1, 2, 3, 4, 5, 6, 7)
        val root = arrayToTreeNode(arr)
        println("result:${maxPathSum(root)}")
        println("Time complexity O(n)")
        println("Space complexity O(h)")
    }

    private fun maxPathSum(root: TreeNode?): Int {
        var maxSum = Int.MIN_VALUE
        fun dfs(root: TreeNode?): Int {
            if (root == null) return 0
            val leftSum = maxOf(0, dfs(root.left))
            val rightSum = maxOf(0, dfs(root.right))
            val priceSum = leftSum + rightSum + root.`val`
            maxSum = maxOf(maxSum, priceSum)
            return maxOf(leftSum, rightSum) + root.`val`
        }
        dfs(root)
        return maxSum
    }

    @Test //38
    fun minStack() {
        println("(easy)min stack")
        val minStack = MinStack().apply {
            push(5)
            push(6)
            push(10)
            push(1)
        }
        println("result:[${minStack.pop()},${minStack.top()},${minStack.getMin()}]")
        println("Time complexity O(1)")
        println("Space complexity O(1)")
    }

    class MinStack {
        private val stack = Stack<Int>()
        private val minStack = Stack<Int>()

        fun push(x: Int): Int {
            stack.push(x)
            if (minStack.isEmpty() || x <= minStack.peek()) {
                minStack.push(x)
            }
            return stack.peek()
        }

        fun pop(): Int {
            val x = stack.pop()
            if (x == minStack.peek()) {
                minStack.pop()
            }
            return stack.peek()
        }

        fun top(): Int {
            return stack.peek()
        }

        fun getMin(): Int {
            return minStack.peek()
        }
    }

    @Test //43
    fun houseRobber() {
        println("(easy)house robber")
        val houses = intArrayOf(2, 7, 9, 3, 1)
        println("result:${rob(houses)}")
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }

    private fun rob(nums: IntArray): Int {
        if (nums.isEmpty()) return 0
        var curr = nums[0]
        var prev = 0

        for (i in 1 until nums.size) {
            val rob = prev + nums[i]
            prev = curr
            curr = maxOf(curr, rob)
        }
        return curr
    }

    @Test //46
    fun coinChange() {
        println("(easy)coin change")
        val coins = intArrayOf(1, 2, 5)
        val amount = 11
        println("result:${coinChange(coins, amount)}")
        println("Time complexity O(n)")
        println("Space complexity O(1)")
    }

    private fun coinChange(coins: IntArray, amount: Int): Int {
        if (amount == 0) return 0
        val dp = IntArray(amount + 1) { amount + 1 }
        dp[0] = 0

        for (i in 1..amount) {
            for (coin in coins) {
                if (i - coin >= 0) {
                    dp[i] = minOf(dp[i], 1 + dp[i - coin])
                }
            }
        }

        return if (dp[amount] == amount + 1) -1 else dp[amount]
    }

    @Test //google
    fun averageSplitsOutput() {
        println("(easy)average splits output")
        val test = mutableListOf<Pair<Int, Int>>()
        test.add(Pair(40, 10))
        test.add(Pair(16, 20))
        test.add(Pair(15, 10))
        test.add(Pair(10, 10))
        test.add(Pair(50, 10))
        test.add(Pair(300, 104123))
        test.add(Pair(22401, 1043))
        test.add(Pair(132049, 10412))
        for(t in test){
            val dataSize = t.first
            val max = t.second
            val splits = findSplits(dataSize, t.second, mutableListOf()).toIntArray()
            val splitCount = splits.size
            if( dataSize % max != 0) {
                val averageDataSize = dataSize / splitCount
                val averageSplits = findSplits(dataSize, averageDataSize, mutableListOf()).toIntArray()
                println("result average splits:${averageDataSize}:[${averageSplits.joinToString { it.toString() }}]")
            } else {
                println("result splits:${splitCount}:[${splits.joinToString { it.toString() }}]")
            }
        }
        println("Time complexity O(1)")
        println("Space complexity O(1)")
    }

    fun findSplits(dataSize: Int, max: Int, currentSplits: MutableList<Int>): List<Int> {
        return if (dataSize <= 0) {
            if (currentSplits.isEmpty()) return listOf(0) else currentSplits
        } else if (dataSize <= max) {
            currentSplits.add(max)
            currentSplits
        } else {
            // dataSize > maxPacketSize
            currentSplits.add(max)
            findSplits(dataSize - max, max, currentSplits)
        }
    }

    @Test
    fun reorderLogFiles() {
        println("(easy)reorder log files")
        val logs = arrayOf("dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero")
        println("result:${reOrderLogFiles(logs).joinToString()}")
        println("Time complexity O(nlogn)")
        println("Space complexity O(n)")
    }

    private fun reOrderLogFiles(logs: Array<String>): Array<String> {
        val letterLogs = mutableListOf<String>()
        val digitLogs = mutableListOf<String>()
        logs.forEach {
            var isDigit = true
            val arr = it.split(" ")
            for (i in 1..<arr.size) {
                if (arr[i].toIntOrNull() == null) {
                    isDigit = false
                    break
                }
            }
            (if (isDigit) digitLogs else letterLogs).add(it)
        }
        digitLogs.onEach {
            letterLogs.add(it)
        }
        return letterLogs.map { it }.toTypedArray()
    }

    fun generateParenthesis() {
        println("(mind)generate parenthesis")
        val n = 3
        println("result:${generateParenthesis(n)}")
        println("Time complexity O(n)")
        println("Space complexity O(n)")
    }

    fun generateParenthesis(n: Int): List<String> {
        if (n == 1) return listOf("()")
        val parenthesis = mapOf(1 to "()")
        for (i in 2..n) {

        }
        return listOf()
    }

    @Test
    fun binarySearch() {
        println("(easy)binary search")
        val nums = intArrayOf(-1, 0, 3, 5, 9, 12)
        val target = 9
        println("result:${search(nums, target)}")
        println("Time complexity O(logn)")
        println("Space complexity O(1)")
    }

    private fun search(nums: IntArray, target: Int): Int {
        var start = 0
        var end = nums.size - 1
        while (start <= end) {
            val mid = start + (end - start) / 2
            when {
                nums[mid] == target -> return mid
                nums[mid] < target -> start = mid + 1
                else -> end = mid - 1
            }
        }
        return -1
    }

    @Test
    fun medianOfTwoSortedArrays() {
        println("(easy)median of two sorted arrays")
        val nums1 = intArrayOf(1, 3)
        val nums2 = intArrayOf(2)
        println("result:${findMedianSortedArrays(nums1, nums2)}")
        println("Time complexity O(m+n)")
        println("Space complexity O(1)")
    }

    private fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray): Double {
        if (nums1.size > nums2.size) {
            return findMedianSortedArrays(nums2, nums1)
        }

        val m = nums1.size
        val n = nums2.size
        var start = 0
        var end = m

        while (start <= end) {
            val partitionX = (start + end) / 2
            val partitionY = (m + n + 1) / 2 - partitionX

            // Handle edge cases for maxLeftX, minRightX, maxLeftY, minRightY
            val maxLeftX = if (partitionX == 0) Int.MIN_VALUE else nums1[partitionX - 1]
            val minRightX = if (partitionX == m) Int.MAX_VALUE else nums1[partitionX]
            val maxLeftY = if (partitionY == 0) Int.MIN_VALUE else nums2[partitionY - 1]
            val minRightY = if (partitionY == n) Int.MAX_VALUE else nums2[partitionY]
            // Check if the partitions are correct
            if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
                if ((m + n) % 2 == 0) {
                    return (maxOf(maxLeftX, maxLeftY) + minOf(minRightX, minRightY)).toDouble() / 2
                } else {
                    return maxOf(maxLeftX, maxLeftY).toDouble()
                }
            } else if (maxLeftX > minRightY) {
                // Move partitionX to the left
                end = partitionX - 1
            } else {
                // Move partitionX to the right
                start = partitionX + 1
            }
        }
        return 0.0
    }

    //Codility Demo Test

    /*
     * given an array A of N integers, returns the smallest positive integer (greater than 0) that does not occur in A.
     * For example, given A = [1, 3, 6`, 4, 1, 2], the function should return 5.
     * Given A = [1, 2, 3], the function should return 4.
     * Given A = [−1, −3], the functio`n should return 1.
     * N is an integer within the range [1..100,000];
    * */
    fun solution1(intArrayOf: IntArray): Int {
        val seen = HashSet<Int>() // Use a HashSet for efficient presence checking

        // Add only positive numbers to the set, ignoring duplicates
        for (num in intArrayOf) {
            if (num > 0) {
                seen.add(num)
            }
        }

        // Iterate through positive integers starting from 1
        var i = 1
        while (true) {
            if (!seen.contains(i)) {
                return i // Found the smallest missing positive integer
            }
            i++
        }
        return 1 // If all positive integers from 1 to N are present
    }

    @Test
    fun mountainArray(){
        println(mountainArrayLength(intArrayOf(1, 2)))        // Output: 2
        println(mountainArrayLength(intArrayOf(2, 5, 3, 2, 4, 1))) // Output: 4
        println(mountainArrayLength(intArrayOf(2, 3, 3, 2, 2, 2, 1)))// Output: 7
        println(mountainArrayLength(intArrayOf(2, 3, 4, 5, 6, 7, 6, 5, 1, 3, 2, 1)))// Output: 12
    }

     private fun mountainArrayLength(a: IntArray) : Int {
         if (a.size < 2) return a.size // no mountian in this case

         var maxLength = 0
         var i = 0
         while (i < a.size - 1) {
             var currentLength = 0
             var peakIndex = i

             // Check for start of mountian
             var isIncrease = false
             var j = i
             while (j < a.size - 1 && a[j] <= a[j + 1]) {
                 isIncrease = true
                 peakIndex++
                 j++
             }
             if (peakIndex == i) {
                 i++
                 continue
             }

             // Check for peak
             var isDecrease = false
             while (j < a.size - 1 && a[j] >= a[j + 1]) {
                 isDecrease = true
                 j++
             }
             if (!isIncrease && !isDecrease) {
                 i++
                 continue
             } else if (!isIncrease) {
                 i = j
             } else if (!isDecrease) {
                 maxLength = maxOf(maxLength, j + 1 - i)
                 i = j
             } else {
                 currentLength = j + 1 - i
                 maxLength = maxOf(maxLength, currentLength)
                 i = j
             }
         }

         return maxLength
     }

    fun solution(N: Int, A: IntArray, B: IntArray): Boolean {
        // Handle cases with no edges
        if (A.isEmpty()) {
            return N <= 1
        }

        val M = A.size
        // Check for self-loops and edge count
        if (M != N - 1) return false
        for (i in A.indices) {
            if (A[i] == B[i]) {
                return false
            }
            if (A[i]<1 || A[i] > N || B[i]<1 || B[i]>N){
                return false
            }
        }
        // Create an adjacency list representation of the graph
        val adj = Array(N + 1) { mutableListOf<Int>() }
        for (i in A.indices) {
            adj[A[i]].add(B[i])
            adj[B[i]].add(A[i])
        }

        // Check if the graph is connected
        val visited = BooleanArray(N + 1)
        fun dfs(u: Int) {
            visited[u] = true
            for (v in adj[u]) {
                if (!visited[v]) {
                    dfs(v)
                }
            }
        }

        // Start DFS from vertex 1
        dfs(1)

        // Check if all reachable vertices are visited
        for (i in 1..N) {
            if (!visited[i] && adj[i].isNotEmpty()) {
                return false
            }
        }

        return true
    }

    @Test
    fun grash() {
        // Test cases
        println(solution(4, intArrayOf(1, 2, 4, 4, 3), intArrayOf(2, 3, 1, 3, 1))) // true
        println(solution(4, intArrayOf(1, 2, 1, 3), intArrayOf(2, 4, 3, 4))) // false
        println(solution(6, intArrayOf(2, 4, 5, 3), intArrayOf(3, 5, 6, 4))) // false
        println(solution(3, intArrayOf(1, 3), intArrayOf(2, 2))) // false
        println(solution(1, intArrayOf(), intArrayOf())) // true
        println(solution(1, intArrayOf(1), intArrayOf(1))) // false
        println(solution(1, intArrayOf(), intArrayOf(1))) // true
        println(solution(3, intArrayOf(1, 2, 3), intArrayOf(2, 3, 1))) // true
        println(solution(3, intArrayOf(1, 2, 3, 2), intArrayOf(2, 3, 1, 1))) // false
    }

    fun solution(T: IntArray, A: IntArray): Int {
        val n = T.size
        val m = A.size

        // Build the graph (adjacency list)
        val adj = Array(n) { mutableListOf<Int>() }
        for (i in 0 until n) {
            if (i != T[i])
                adj[T[i]].add(i)
        }

        // Function to get all prerequisites for a skill
        fun getPrerequisites(skill: Int, learned: MutableSet<Int>) {
            if(skill != 0){
                val parent = T[skill]
                if (!learned.contains(parent)) {
                    learned.add(parent)
                    getPrerequisites(parent, learned)
                }

            }


        }

        val learnedSkills = mutableSetOf<Int>()
        for (skill in A) {
            if (!learnedSkills.contains(skill)) {
                learnedSkills.add(skill)
                getPrerequisites(skill, learnedSkills)
            }

        }


        return learnedSkills.size
    }

    @Test
    fun main1() {
        // Test cases
        println(solution(intArrayOf(0, 0, 1, 1), intArrayOf(2))) // Output: 3
        println(solution(intArrayOf(0, 0, 0, 0, 2, 3, 3), intArrayOf(2, 5, 6))) // Output: 5
        println(solution(intArrayOf(0, 0, 1, 2), intArrayOf(1, 2))) // Output: 3
        println(solution(intArrayOf(0,0,0,0,1,2,3,4), intArrayOf(5,6,7))) // Output: 7
        println(solution(intArrayOf(0,0,0,0,1,2,3,4), intArrayOf(0))) // Output: 1
    }

    fun solution(N: Int): Int {
        if (N == 0) return 0
        if (N == 1) return 1

        val sequence = mutableListOf(0, 1)
        val seen = mutableMapOf<Pair<Int, Int>, Int>() // Map to store pairs of previous numbers and their index

        var n = 2
        while (n <= N) {
            val prev1 = sequence[n - 1]
            val prev2 = sequence[n - 2]
            var sum = 0
            var temp = prev1
            while (temp > 0) {
                sum += temp % 10
                temp /= 10
            }
            temp = prev2
            while (temp > 0) {
                sum += temp % 10
                temp /= 10
            }

            sequence.add(sum)

            // Check for cycle
            val pair = Pair(prev1, prev2)
            if (seen.containsKey(pair)) {
                val cycleStart = seen[pair]!!
                val cycleLength = n - cycleStart
                val remaining = N - cycleStart
                val indexInCycle = remaining % cycleLength
                return sequence[cycleStart + indexInCycle]
            } else {
                seen[pair] = n
            }

            n++
        }

        return sequence[N]
    }

    @Test
    fun main() {
        println(solution(0)) // 0
        println(solution(1)) // 1
        println(solution(2)) // 1
        println(solution(6)) // 8
        println(solution(10)) // 10
        println(solution(11)) // 8
        println(solution(12)) // 9
        println(solution(13)) // 7
        println(solution(14))//10
        println(solution(15))//8
        println(solution(16))//9
        println(solution(17))//7
        println(solution(18))//10
        println(solution(19))//8
        println(solution(100))
        println(solution(10000))
    }


}