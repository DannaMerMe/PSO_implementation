class ValidationError(Exception):
    """Excepción personalizada para errores de validación."""
    pass


def validate_integer_parameter(value, name):
    """
    Valida que un parámetro sea un entero positivo.
    """
    try:
        numero = int(value)
        if numero <= 0:
            raise ValidationError(f"{name} debe ser un número entero positivo.")
        return numero
    except ValueError:
        raise ValidationError(f"Error: '{value}' no es válido. {name} debe ser un número entero.")


def validate_float_parameter(value, name):
    """
    Valida que un parámetro sea un float válido.
    """
    try:
        return float(value)
    except ValueError:
        raise ValidationError(f"Error: '{value}' no es válido. {name} debe ser un número.")
