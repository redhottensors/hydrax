#include <stddef.h>

#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    PyObject *buffer;
} TrackedBuffer;

static PyObject *TrackedBuffer_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    PyObject *buffer;

    static char *argnames[] = {"buffer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:TrackedBuffer", argnames, &buffer)) {
        return NULL;
    }

    if (!PyObject_CheckBuffer(buffer)) {
        PyErr_Format(PyExc_TypeError, "a bytes-like object is required, not '%.100s'", Py_TYPE(buffer)->tp_name);
        return NULL;
    }

    TrackedBuffer *self = (TrackedBuffer *)subtype->tp_alloc(subtype, 0);
    if (!self) {
        return NULL;
    }

    Py_INCREF(buffer);
    self->buffer = buffer;
    return (PyObject *)self;
}

static int TrackedBuffer_traverse(PyObject *obj, visitproc visit, void *arg) {
    TrackedBuffer *self = (TrackedBuffer *)obj;
    Py_VISIT(self->buffer);
    return 0;
}

static int TrackedBuffer_clear(PyObject *obj) {
    // We can't clear self->buffer here, as there may still be a reference to it through us.
    return 0;
}

static void TrackedBuffer_dealloc(PyObject *obj) {
    TrackedBuffer *self = (TrackedBuffer *)obj;

    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->buffer);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int TrackedBuffer_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
    TrackedBuffer *self = (TrackedBuffer *)obj;

    if (!self->buffer) {
        PyErr_SetString(PyExc_BufferError, "TrackedBuffer is uninitialized");
        view->obj = NULL;
        return -1;
    }

    if (PyObject_GetBuffer(self->buffer, view, flags)) {
        return -1;
    }

    Py_INCREF(self);
    view->obj = (PyObject *)self;

    PyObject *result = PyObject_CallMethod((PyObject *)self, "_ref", NULL);
    if (!result) {
        PyErr_Clear();

        view->obj = self->buffer;
        PyBuffer_Release(view);
        Py_DECREF(self);

        PyErr_SetString(PyExc_BufferError, "TrackedBuffer _ref failed");
        view->obj = NULL;
        return -1;
    }

    Py_DECREF(result);
    return 0;
}

static void TrackedBuffer_releasebuffer(PyObject *obj, Py_buffer *view) {
    TrackedBuffer *self = (TrackedBuffer *)obj;

    view->obj = self->buffer;
    PyBuffer_Release(view);

    // PyBuffer_Release will DECREF view->obj, so we must not do so here.
    view->obj = (PyObject *)self;

    PyObject *result = PyObject_CallMethod((PyObject *)self, "_deref", NULL);
    if (!result) {
        PyErr_Print();
        return;
    }

    Py_DECREF(result);
}

static PyObject *TrackedBuffer_repr(PyObject *obj) {
    TrackedBuffer *self = (TrackedBuffer *)obj;

    if (!self->buffer) {
        return PyUnicode_FromFormat(
            "<%s object at %p buffer=None>",
            Py_TYPE(self)->tp_name, self
        );
    }

    // Get the repr of the buffer
    PyObject *repr = PyObject_Repr(self->buffer);
    if (!repr) {
        PyErr_Clear();

        return PyUnicode_FromFormat(
            "<%s object at %p buffer=<%s object at %p>>",
            Py_TYPE(self)->tp_name, self,
            Py_TYPE(self->buffer)->tp_name, self->buffer
        );
    }

    PyObject* result = PyUnicode_FromFormat(
        "<%s object at %p buffer=%U>",
        Py_TYPE(self)->tp_name, self,
        repr
    );
    Py_DECREF(repr);

    return result;
}

static PyObject *pass(PyObject *self, PyObject *args) {
    return Py_None;
}

static PyBufferProcs TrackedBuffer_bufferprocs = {
    .bf_getbuffer = TrackedBuffer_getbuffer,
    .bf_releasebuffer = TrackedBuffer_releasebuffer,
};

static PyMethodDef TrackedBuffer_methods[] = {
    {"_ref", pass, METH_NOARGS, PyDoc_STR("Called when the tracked buffer is referenced via this object.")},
    {"_deref", pass, METH_NOARGS, PyDoc_STR("Called when the tracked buffer is dereferenced via this object.")},
    {}
};

static PyMemberDef TrackedBuffer_members[] = {
    {"buffer", T_OBJECT_EX, offsetof(TrackedBuffer, buffer), READONLY, PyDoc_STR("The tracked buffer object. Using this object bypasses tracking.")},
    {}
};

#ifndef Py_TPFLAGS_IMMUTABLETYPE
#define Py_TPFLAGS_IMMUTABLETYPE 0
#endif

static PyTypeObject TrackedBuffer_type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    .tp_name = "hydrax._trackedbuffer.TrackedBuffer",
    .tp_basicsize = sizeof(TrackedBuffer),
    .tp_itemsize = 0,
    .tp_dealloc = TrackedBuffer_dealloc,
    .tp_repr = TrackedBuffer_repr,
    .tp_getattro = PyObject_GenericGetAttr,
    .tp_setattro = PyObject_GenericSetAttr,
    .tp_as_buffer = &TrackedBuffer_bufferprocs,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = PyDoc_STR("Buffer protocol wrapper which provides callbacks when the tracked buffer is referenced or dereferenced."),
    .tp_traverse = TrackedBuffer_traverse,
    .tp_clear = TrackedBuffer_clear,
    .tp_methods = TrackedBuffer_methods,
    .tp_members = TrackedBuffer_members,
    .tp_new = TrackedBuffer_new,
};

static PyModuleDef TrackedBuffer_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_trackedbuffer",
    .m_doc = PyDoc_STR("Provides the TrackedBuffer class for tracking buffer protocol reference operations."),
    .m_size = 0,
};

PyMODINIT_FUNC PyInit__trackedbuffer(void) {
    if (PyType_Ready(&TrackedBuffer_type)) {
        return NULL;
    }

    PyObject* module = PyModule_Create(&TrackedBuffer_module);
    if (!module) {
        return NULL;
    }

    Py_INCREF(&TrackedBuffer_type);
    if (PyModule_AddObject(module, "TrackedBuffer", (PyObject *)&TrackedBuffer_type)) {
        Py_DECREF(&TrackedBuffer_type);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
