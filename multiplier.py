# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--lp_file")
# parser.add_argument("--multiplier", default=1000000)
#
# args = parser.parse_args()
# LP_FILE = args.lp_file
# MULTIPLIER = args.multiplier

def my_is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def multiply_lp(LP_FILE_IN,LP_FILE_OUT, MULTIPLIER):
	linear_program = ''
	f = open(LP_FILE_IN, "r")
	finished = False
	start = True
	for line in f.readlines():
		pieces = line.rstrip().split(" ")
		if not finished and start:
			for i, piece in enumerate(pieces):
				if my_is_number(piece):
					pieces[i] = str(float(pieces[i])*MULTIPLIER)
				if i > 0 and not my_is_number(pieces[i-1]):
					if len(piece)>0:
						if piece[0] == 'x' or piece[0] == 'y':
							pieces[i] = str(str(MULTIPLIER)+" "+pieces[i])
		#if 'Binary' in line:
		if 'Subject' in line:
			start = True
		if 'Bounds' in line:
			finished = True
		linear_program += ' '.join(pieces)+"\n"
	text_file = open(LP_FILE_OUT, "w")
	text_file.write(linear_program)
	text_file.close()