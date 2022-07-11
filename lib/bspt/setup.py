try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
import numpy
numpy_include_dir = numpy.get_include()


bspt_module = Extension(
    'bspt', 
    sources=[
        'bspt.pyx'
    ]
)

# Gather all extension modules
ext_modules = [
    bspt_module,
]


def set_builtin(name, value):
    if isinstance(__builtins__, dict):
        __builtins__[name] = value
    else:
        setattr(__builtins__, name, value)


class build_ext_subclass(build_ext):
    def build_extensions(self):
        comp = self.compiler.compiler_type
        if comp in ('unix', 'cygwin', 'mingw32'):
            # Check if build is with OpenMP
            extra_compile_args = ['-std=c99', '-O3', '-fopenmp']
            extra_link_args=['-lgomp']
        elif comp == 'msvc':
            extra_compile_args = ['/Ox']
            extra_link_args = []
            extra_compile_args.append('/openmp')
        else:
            # Add support for more compilers here
            raise ValueError('Compiler flags undefined for %s. Please modify setup.py and add compiler flags'
                             % comp)
        self.extensions[0].extra_compile_args = extra_compile_args
        self.extensions[0].extra_link_args = extra_link_args
        build_ext.build_extensions(self)

    def finalize_options(self):
        '''
        In order to avoid premature import of numpy before it gets installed as a dependency
        get numpy include directories during the extensions building process
        http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
        '''
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        set_builtin('__NUMPY_SETUP__', False)
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': build_ext_subclass
    }
)

# run: python setup.py build_ext --inplace