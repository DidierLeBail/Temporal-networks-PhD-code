import os

class Char_subst:
	def __init__(self):
		self.chars = ['#', ',', '+', '-', '*', ':', '=', '<', '>']
		self.char_to_subst = {
			'#': self.replace_hashtag,
			',': self.replace_hashtag,
			'+': self.replace_hashtag,
			'-': self.replace_hashtag,
			'*': self.replace_hashtag,
			':': self.replace_hashtag,
			'=': self.replace_hashtag,
			'<': self.replace_hashtag,
			'>': self.replace_hashtag
		}
	
	@staticmethod
	def replace_1(list_letters, pos, char=''):
		left = ' '
		right = ' '
		if list_letters[pos + 1] in ['\n', ' ', char, '=']:
			right = ''
		if list_letters[pos - 1] in [' ', char]:
			left = ''
		list_letters[pos] = left + list_letters[pos] + right

	@staticmethod
	def replace_hashtag(list_letters, pos):
		if list_letters[pos + 1] in ['\n', ' ', '#']:
			return None
		list_letters[pos] += ' '

	@staticmethod
	def replace_plus(*args):
		return Char_subst.replace_1(*args, chr = '+')

	@staticmethod
	def replace_times(list_letters, pos):
		left = ' '
		right = ' '
		if list_letters[pos + 1] in ['\n', ' ', '*', '=', '(']:
			right = ''
		if list_letters[pos - 1] in [' ', '=', '*', '(']:
			left = ''
		list_letters[pos] = left + list_letters[pos] + right

	@staticmethod
	def replace_colon(*args):
		return Char_subst.replace_1(*args, chr = ':')

	@staticmethod
	def replace_minus(list_letters, pos):
		if list_letters[pos + 1] in ['\n', ' ', '=']:
			right = ''
		else:
			right = ' '
		left = ' '
		if list_letters[pos - 1] == ' ':
			left = ''
		if len(list_letters[:pos]) >= 2:
			if ''.join(list_letters[pos - 2: pos]) == '1e':
				left = ''
		list_letters[pos] = left + list_letters[pos] + right

	@staticmethod
	def replace_l(*args):
		return Char_subst.replace_1(*args, chr = '<')

	@staticmethod
	def replace_g(*args):
		return Char_subst.replace_1(*args, chr = '>')

	@staticmethod
	def replace_equal(list_letters, pos):
		left = ' '
		right = ' '
		if list_letters[pos + 1] in ['\n', ' ', '=']:
			right = ''
		if list_letters[pos - 1] in [' ', '=', '<', '>', '+', '-', '%', '/', '*']:
			left = ''
		list_letters[pos] = left + list_letters[pos] + right

	def __call__(self, char, *args, **kwargs):
		if char in self.char_to_subst:
			return self.char_to_subst[char](*args, **kwargs)
		else:
			return None

def correct_fmt(filename):
	"""add a space after and / or before the following characters:
	- # except if followed by a #
	- ,
	- + except if followed by a =
	- * except if followed by a = or *
	- - except if followed by a = or follows a 1e
	- : (in dict or slices)
	- =
	- >
	- <
	"""
	subst = Char_subst()
	lines = []
	with open(filename, "r") as f:
		for line in f.readlines():
			list_letters = list(line)
			for pos, char in enumerate(line):
				subst(char, list_letters, pos)
			lines.append(''.join(list_letters))
	with open(filename, 'w') as f:
		f.writelines(lines)

filename = "papers/ADM/test.py"
correct_fmt(filename)
