{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
    {% block modules %}
    {% if modules %}
    .. autosummary::
       :toctree:
       :template: custom-module-template.rst
       :recursive:
    {% for item in modules %}
       {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. autosummary::
      :toctree:
      :template: custom-class-template.rst
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

