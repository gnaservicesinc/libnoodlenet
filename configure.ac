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

# Determine host OS for optional Apple Accelerate
AC_CANONICAL_HOST
ACCELERATE_LIBS=""
case $host_os in
  darwin*)
    ACCELERATE_LIBS="-Wl,-framework,Accelerate"
    ;;
esac
AC_SUBST([ACCELERATE_LIBS])

# Optional Thin LTO setup for macOS
case $host_os in
  darwin*)
    AC_MSG_CHECKING([for compiler vendor (Apple clang vs LLVM clang)])
    cc_version_str=`$CC --version 2>/dev/null | head -n 1`
    AC_MSG_RESULT([$cc_version_str])

    # LTO cache path at project root (build directory)
    LTO_CACHE_PATH="$PWD/ltocache"
    AC_SUBST([LTO_CACHE_PATH])

    # Enable thin LTO by default on macOS; allow override via --disable-thin-lto
    AC_ARG_ENABLE([thin-lto],
      [AS_HELP_STRING([--disable-thin-lto],[Disable Thin LTO on macOS])],
      [],
      [enable_thin_lto=yes])

    if test "x$enable_thin_lto" = "xyes"; then
      AC_MSG_NOTICE([Enabling Thin LTO (thin) with cache at $LTO_CACHE_PATH])
      CFLAGS="$CFLAGS -flto=thin -fPIC -Wl,-cache_path_lto,$LTO_CACHE_PATH"
      CXXFLAGS="$CXXFLAGS -flto=thin -fPIC -Wl,-cache_path_lto,$LTO_CACHE_PATH"
      LDFLAGS="$LDFLAGS -flto=thin -fPIC -L/usr/local/lib -Wl,-rpath,/usr/local/lib -Wl,-cache_path_lto,$LTO_CACHE_PATH"

      case "$cc_version_str" in
        *"Apple clang"*)
          # Prefer ld-classic from Xcode or Command Line Tools
          AC_MSG_CHECKING([for ld-classic linker])
          for p in \
            /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ld-classic \
            /Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ld-classic \
            /Library/Developer/CommandLineTools/usr/bin/ld-classic; do
            if test -x "$p"; then
              LD="$p"
              break
            fi
          done
          if test "x$LD" != "x"; then
            AC_MSG_RESULT([$LD])
          else
            AC_MSG_RESULT([not found])
            AC_MSG_WARN([ld-classic not found; Thin LTO may not link correctly])
          fi
          ;;
        *"clang version "*)
          # LLVM clang: use InstalledDir/ld64.lld
          AC_MSG_CHECKING([for ld64.lld linker])
          instdir=`$CC --version 2>/dev/null | sed -n 's/^InstalledDir:[ ]*\(.*\)/\1/p' | head -n 1`
          if test "x$instdir" != "x" && test -x "$instdir/ld64.lld"; then
            LD="$instdir/ld64.lld"
            AC_MSG_RESULT([$LD])
          else
            AC_MSG_RESULT([not found])
            AC_MSG_WARN([ld64.lld not found in InstalledDir; Thin LTO may not link correctly])
          fi
          ;;
      esac
    fi
    ;;
esac

# Optional Polly + OpenMP when using LLVM clang (not Apple clang)
AC_ARG_ENABLE([polly-omp],
  [AS_HELP_STRING([--enable-polly-omp],[Enable Polly automatic OpenMP parallelization when using LLVM clang])],
  [],
  [enable_polly_omp=auto])
AC_ARG_ENABLE([polly-vectorize],
  [AS_HELP_STRING([--enable-polly-vectorize],[Also enable Polly vectorizer (stripmine)])],
  [],
  [enable_polly_vectorize=no])
AC_ARG_WITH([polly-threads],
  [AS_HELP_STRING([--with-polly-threads=N],[Set Polly OpenMP threads (default 12)])],
  [polly_threads="$withval"],
  [polly_threads=12])

OPENMP_LIBS=""
case "$enable_polly_omp" in
  yes|auto)
    case "$cc_version_str" in
      *"Apple clang"*)
        if test "x$enable_polly_omp" = "xyes"; then
          AC_MSG_WARN([--enable-polly-omp requested but compiler is Apple clang; skipping Polly/OpenMP flags])
        fi
        ;;
      *"clang version "*)
        # Prefer LLVM clang (non-Apple). Try to enable Polly and OpenMP if libomp is present.
        AC_MSG_CHECKING([for OpenMP header and library in /usr/local])
        if test -f "/usr/local/include/omp.h" -a -f "/usr/local/lib/libomp.dylib"; then
          AC_MSG_RESULT([found])
          AC_MSG_NOTICE([Enabling Polly (+parallel) with $polly_threads threads])
          CFLAGS="$CFLAGS -I/usr/local/include -D_THREAD_SAFE -Xclang -fopenmp -mllvm -polly -mllvm -polly-parallel -mllvm -polly-omp-backend=LLVM -mllvm -polly-num-threads=$polly_threads"
          CXXFLAGS="$CXXFLAGS -I/usr/local/include -D_THREAD_SAFE -Xclang -fopenmp -mllvm -polly -mllvm -polly-parallel -mllvm -polly-omp-backend=LLVM -mllvm -polly-num-threads=$polly_threads"
          if test "x$enable_polly_vectorize" = "xyes"; then
            CFLAGS="$CFLAGS -mllvm -polly-vectorizer=stripmine"
            CXXFLAGS="$CXXFLAGS -mllvm -polly-vectorizer=stripmine"
          fi
          OPENMP_LIBS="-L/usr/local/lib -lomp"
          # Ensure CXX matches the LLVM clang toolchain (avoid Apple g++)
          instdir=`$CC --version 2>/dev/null | sed -n 's/^InstalledDir:[ ]*\(.*\)/\1/p' | head -n 1`
          if test "x$instdir" = "x"; then
            ccpath=`command -v $CC 2>/dev/null`
            instdir=`dirname "$ccpath"`
          fi
          if test -x "$instdir/clang++"; then
            AC_MSG_NOTICE([Using matching LLVM C++ compiler: $instdir/clang++])
            CXX="$instdir/clang++"
          elif test -x "/usr/local/bin/clang++"; then
            AC_MSG_NOTICE([Using LLVM C++ compiler: /usr/local/bin/clang++])
            CXX="/usr/local/bin/clang++"
          else
            AC_MSG_WARN([Could not find matching clang++; current CXX is $CXX])
          fi
          AC_SUBST([CXX])
        else
          AC_MSG_RESULT([not found])
          if test "x$enable_polly_omp" = "xyes"; then
            AC_MSG_WARN([Requested --enable-polly-omp but OpenMP headers/libs not found in /usr/local; skipping])
          fi
        fi
        ;;
      *)
        # Unknown; skip Polly
        if test "x$enable_polly_omp" = "xyes"; then
          AC_MSG_WARN([--enable-polly-omp requested but compiler vendor unknown; skipping])
        fi
        ;;
    esac
    ;;
  no)
    ;;
esac
AC_SUBST([OPENMP_LIBS])

# Configure files to generate
AC_CONFIG_FILES([
 Makefile
 libnoodlenet.pc
])

AC_OUTPUT
