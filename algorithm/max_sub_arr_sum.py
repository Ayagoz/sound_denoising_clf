def max_sub_array(nums):
	"""
	:type nums: List[int]
	:rtype: int
	"""
	copy = nums[:]

	for i in range(1, len(copy)):
	    copy[i] = max(copy[i], copy[i - 1] + nums[i])

	return max(copy)

if __name__ == "__main__":
	in_ = input()
	try:
		nums = list(map(float, in_.split(',')))
	except:
		with open(in_, 'r') as f:
			nums = list(map(float, f.read().split(',')))
	
	print(f'Max sub-array of sequence is {max_sub_array(nums)}')

