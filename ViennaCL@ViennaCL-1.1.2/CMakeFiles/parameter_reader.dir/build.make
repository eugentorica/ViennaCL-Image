# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sanda/Desktop/ViennaCL-1.1.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sanda/Desktop/ViennaCL-1.1.2

# Include any dependencies generated for this target.
include CMakeFiles/parameter_reader.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/parameter_reader.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/parameter_reader.dir/flags.make

CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o: CMakeFiles/parameter_reader.dir/flags.make
CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o: examples/parameters/parameter_reader.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sanda/Desktop/ViennaCL-1.1.2/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o -c /home/sanda/Desktop/ViennaCL-1.1.2/examples/parameters/parameter_reader.cpp

CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sanda/Desktop/ViennaCL-1.1.2/examples/parameters/parameter_reader.cpp > CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.i

CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sanda/Desktop/ViennaCL-1.1.2/examples/parameters/parameter_reader.cpp -o CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.s

CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o.requires:
.PHONY : CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o.requires

CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o.provides: CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o.requires
	$(MAKE) -f CMakeFiles/parameter_reader.dir/build.make CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o.provides.build
.PHONY : CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o.provides

CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o.provides.build: CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o
.PHONY : CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o.provides.build

CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o: CMakeFiles/parameter_reader.dir/flags.make
CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o: external/pugixml/src/pugixml.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sanda/Desktop/ViennaCL-1.1.2/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o -c /home/sanda/Desktop/ViennaCL-1.1.2/external/pugixml/src/pugixml.cpp

CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sanda/Desktop/ViennaCL-1.1.2/external/pugixml/src/pugixml.cpp > CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.i

CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sanda/Desktop/ViennaCL-1.1.2/external/pugixml/src/pugixml.cpp -o CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.s

CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o.requires:
.PHONY : CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o.requires

CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o.provides: CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o.requires
	$(MAKE) -f CMakeFiles/parameter_reader.dir/build.make CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o.provides.build
.PHONY : CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o.provides

CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o.provides.build: CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o
.PHONY : CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o.provides.build

# Object files for target parameter_reader
parameter_reader_OBJECTS = \
"CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o" \
"CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o"

# External object files for target parameter_reader
parameter_reader_EXTERNAL_OBJECTS =

parameter_reader: CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o
parameter_reader: CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o
parameter_reader: CMakeFiles/parameter_reader.dir/build.make
parameter_reader: CMakeFiles/parameter_reader.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable parameter_reader"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/parameter_reader.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/parameter_reader.dir/build: parameter_reader
.PHONY : CMakeFiles/parameter_reader.dir/build

CMakeFiles/parameter_reader.dir/requires: CMakeFiles/parameter_reader.dir/examples/parameters/parameter_reader.cpp.o.requires
CMakeFiles/parameter_reader.dir/requires: CMakeFiles/parameter_reader.dir/external/pugixml/src/pugixml.cpp.o.requires
.PHONY : CMakeFiles/parameter_reader.dir/requires

CMakeFiles/parameter_reader.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/parameter_reader.dir/cmake_clean.cmake
.PHONY : CMakeFiles/parameter_reader.dir/clean

CMakeFiles/parameter_reader.dir/depend:
	cd /home/sanda/Desktop/ViennaCL-1.1.2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sanda/Desktop/ViennaCL-1.1.2 /home/sanda/Desktop/ViennaCL-1.1.2 /home/sanda/Desktop/ViennaCL-1.1.2 /home/sanda/Desktop/ViennaCL-1.1.2 /home/sanda/Desktop/ViennaCL-1.1.2/CMakeFiles/parameter_reader.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/parameter_reader.dir/depend

