AC_INIT([libnoodlenet], [1.0], [info@example.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CC
AC_PROG_CXX
AM_PROG_AR
LT_INIT
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])

# Check for required libraries
AC_CHECK_HEADERS([stdio.h stdlib.h string.h math.h])

# Configure files to generate
AC_CONFIG_FILES([
 Makefile
 libnoodlenet.pc
])

AC_OUTPUT
