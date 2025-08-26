import io
# pickle is not secure, but but this whole file is a wrapper to make it
# possible to mitigate the primary risk of code injection via pickle.
import pickle  # nosec B403

# These are the base classes that are generally serialized by the ZeroMQ IPC.
# If a class is needed by ZMQ routinely it should be added here. If
# it is only needed in a single instance the class can be added at runtime
# using register_approved_ipc_class.
BASE_CLASSES = {'gr00t.data.dataset': ['ModalityConfig'], 'numpy.core.multiarray': ['_reconstruct'], 'numpy': ['ndarray', 'dtype']}

class Unpickler(pickle.Unpickler):

    def __init__(self, *args, approved_imports={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.approved_imports = approved_imports

    # only import approved classes, this is the security boundary.
    def find_class(self, module, name):
        if name not in self.approved_imports.get(module, []):
            # If this is triggered when it shouldn't be, then the module
            # and class should be added to the approved_imports. If the class
            # is being used as part of a routine scenario, then it should be added
            # to the appropriate base classes above.

            # Uncomment to update approve list.
            # BASE_CLASSES[module].append(name)
            # print(BASE_CLASSES)
            raise ValueError(f"Import {module} | {name} is not allowed")
        return super().find_class(module, name)


# these are taken from the pickle module to allow for this to be a drop in replacement
# source: https://github.com/python/cpython/blob/3.13/Lib/pickle.py
# dump and dumps are just aliases because the serucity controls are on the deserialization
# side. However they are included here so that in the future if a more secure serialization
# soliton is identified, it can be added with less impact to the rest of the application.
dump = pickle.dump  # nosec B301
dumps = pickle.dumps  # nosec B301


def load(file,
         *,
         fix_imports=True,
         encoding="ASCII",
         errors="strict",
         buffers=None,
         approved_imports=BASE_CLASSES):
    return Unpickler(file,
                     fix_imports=fix_imports,
                     buffers=buffers,
                     encoding=encoding,
                     errors=errors,
                     approved_imports=approved_imports).load()


def loads(s,
          /,
          *,
          fix_imports=True,
          encoding="ASCII",
          errors="strict",
          buffers=None,
          approved_imports=BASE_CLASSES):
    if isinstance(s, str):
        raise TypeError("Can't load pickle from unicode string")
    file = io.BytesIO(s)
    return Unpickler(file,
                     fix_imports=fix_imports,
                     buffers=buffers,
                     encoding=encoding,
                     errors=errors,
                     approved_imports=approved_imports).load()