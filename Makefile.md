# Makefile
## Makefile Syntax
```makefile
targets: prerequisites 
command 
command 
command

blah: blah.c 
cc blah.c -o blah
```
When we run `make` again, the following set of steps happens:
- The first target is selected, because the first target is the default target
- This has a prerequisite of `blah.c`
- Make decides if it should run the `blah` target. It will only run if `blah` doesn't exist, or `blah.c` is _newer than_ `blah`
Note that `clean` is doing two new things here:
- It's a target that is not first (the default), and not a prerequisite. That means it'll never run unless you explicitly call `make clean`
- It's not intended to be a filename. If you happen to have a file named `clean`, this target won't run, which is not what we want. See `.PHONY` later in this tutorial on how to fix this
You'll typically want to use `:=`
```makefile
files := file1 file2 
some_file: $(files) 
echo "Look at this variable: " $(files) 
touch some_file
```
Reference variables using either `${}` or `$()`
## The all target

Making multiple targets and you want all of them to run? Make an `all` target. Since this is the first rule listed, it will run by default if `make` is called without specifying a target.
```
all: one two three

one:
	touch one
two:
	touch two
three:
	touch three

clean:
	rm -f one two three
```