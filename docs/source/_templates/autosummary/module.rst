{{ name | escape | underline}}

.. List the submodules

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
{% set full_item = fullname + '.' + item.split('.')[-1] %}
{% if ('.plugins.' not in full_item or full_item == 'modelopt.torch.opt.plugins.huggingface') and full_item != 'modelopt.torch.quantization.backends.fp8_per_tensor_gemm' %}
   {{ full_item }}
{% endif %}
{%- endfor %}
{% endif %}
{% endblock %}

.. Autodoc anything defined in the module itself

   TODO: WE DON'T USE THIS OPTION RIGHT NOW BUT WE CAN REACTIVATE IF WANTED
   We use :ignore-module-all: so sphinx does not document the same module twice, even if it is reimported
   For reimports that should be documented somewhere other than where they are defined, the re-imports
   __module__ should be manually overridden -- i.e. in the ``__init__.py`` which contains ``from xxx import YYY``,
   add in ``YYY.__module__ = __name__``.

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:

   .. Also show members without docstrings. Only members from __all__ are considered as per conf.py
   .. Ideally we should add docstrings for these members.


   .. Overview table of available classes in the module
   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   .. Overview table of available functions in the module
   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
