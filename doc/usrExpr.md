# usrExpr
## Overview

The module implements a small, restricted expression evaluator named `UserExpression`. It is designed to evaluate simple arithmetic expressions provided as strings against a nested dictionary context, while rejecting unsupported Python syntax and preventing arbitrary code execution.  

The implementation uses Python's `ast` module to parse the expression into an abstract syntax tree and then walks that tree manually. Only a tightly controlled subset of AST node types and operators is accepted.  

## Module contents

The script contains three main parts:  

- `_ALLOWED_BINOPS`: mapping of allowed binary AST operators to Python operator functions.  
- `_ALLOWED_UNARYOPS`: mapping of allowed unary AST operators to Python operator functions.  
- `UserExpression`: the evaluator class.  

## Allowed operations

The evaluator supports these binary operations:  

| AST node | Symbol | Python operator |
| --- | --- | --- |
| `ast.Add` | `+` | `operator.add` |
| `ast.Sub` | `-` | `operator.sub` |
| `ast.Mult` | `*` | `operator.mul` |
| `ast.Div` | `/` | `operator.truediv` |
| `ast.Pow` | `**` | `operator.pow` |

The evaluator supports these unary operations:  

| AST node | Symbol | Python operator |
| --- | --- | --- |
| `ast.UAdd` | unary `+` | `operator.pos` |
| `ast.USub` | unary `-` | `operator.neg` |

No function calls, indexing, comprehensions, boolean logic, comparisons, lambdas, or statements are supported by this implementation. Any unsupported node type raises `TypeError`.  

## `UserExpression` class

### Purpose

`UserExpression` encapsulates a parsed user expression and provides a `value(context)` method that evaluates it against a supplied context dictionary. The result is always converted to `float`.  

This makes the class suitable for configuration-driven formulas such as inlet expressions, dependent variables, or simple user-defined scalar relations.  

### Constructor

```python
UserExpression(expression: str)
```

The constructor stores the original expression string and parses it immediately using `ast.parse(expression, mode="eval")`. Parsing at construction time means syntax errors are detected early, before any evaluation call is made.  

Stored attributes:  

- `self.expression`: original expression string.  
- `self._tree`: parsed AST in evaluation mode.  

### Public method

#### `value(context: dict) -> float`

This is the main public API. It evaluates the root expression node using the private recursive evaluator and returns the final result as a floating-point number.  

```python
expr = UserExpression("2*x + 1")
result = expr.value({"x": 3})
# result == 7.0
```

## Internal evaluation flow

### `_eval_node(node, context)`

This method recursively evaluates one AST node at a time. It acts as a dispatcher that recognizes the node type and applies the correct evaluation rule.  

Supported node categories are:  

- `ast.Constant` for numeric literals.  
- `ast.BinOp` for binary arithmetic.  
- `ast.UnaryOp` for unary plus and minus.  
- `ast.Name` for top-level context variables.  
- `ast.Attribute` for dotted dictionary paths.  

Any other node type triggers `TypeError` with the dumped AST node for easier debugging.  

### Constant handling

If the node is `ast.Constant`, only `int` and `float` values are accepted. Any other constant type, such as strings or booleans, raises `TypeError`.  

This means expressions like `3.14`, `2`, or `-5` are valid, while `'abc'` or `True` are rejected.  

### Binary operations

For `ast.BinOp`, the implementation checks whether the operator type exists in `_ALLOWED_BINOPS`. If it is allowed, the left and right operands are evaluated recursively and then combined with the mapped Python operator function.  

This supports expressions such as:  

- `a + b`  
- `a - b`  
- `a * b`  
- `a / b`  
- `a ** b`  

If the operator is not in the whitelist, the code raises `TypeError` with a message such as `Operator not allowed: Mod`.  

### Unary operations

For `ast.UnaryOp`, the evaluator checks `_ALLOWED_UNARYOPS`. It supports unary plus and unary minus only, then recursively evaluates the operand and applies the mapped operator function.  

This allows expressions such as `-x`, `+x`, or `-(a + b)`.  

### Name resolution

If the node is `ast.Name`, the evaluator delegates to `_resolve_name(name, context)`. This method simply looks up the identifier in the top-level context dictionary.  

If the name is missing, the method raises `KeyError` with `Unknown name: <name>`.  

### Attribute-chain resolution

If the node is `ast.Attribute`, the evaluator interprets it as a dotted dictionary path and delegates to `_resolve_attribute_chain(node, context)`.  

The method walks backward through nested `ast.Attribute` nodes until it reaches an `ast.Name`, collecting the full path. For an expression like `inlet.specie.so2`, it constructs the list `['inlet', 'specie', 'so2']` and then descends through nested dictionaries in the supplied context.  

Resolution rules are strict:  

- The chain must start from a simple name.  
- Every intermediate value must be a dictionary.  
- Every path segment must exist.  

If the chain starts from something other than a plain name, the code raises `TypeError` with `Only simple dotted paths are allowed`. If traversal encounters a non-dictionary or a missing key, it raises `KeyError`.  

## Supported expression style

The module is best suited to expressions like these:  

```python
"0.5"
"a + b"
"2 * x - 1"
"inlet.specie.so2 * 0.5"
"1.0 - inlet.specie.so2 - expr1"
"-(x ** 2)"
```

These expressions work because they rely only on numeric constants, arithmetic operators, plain names, and dotted dictionary access.  

## Unsupported syntax

The following kinds of expressions are rejected by design:  

- Function calls such as `sin(x)` or `max(a, b)`.  
- Indexing such as `a[0]` or `dict['key']`.  
- Comparisons such as `x > 0`.  
- Boolean logic such as `a and b`.  
- Conditional expressions such as `x if cond else y`.  
- Method calls or object access outside dictionary-style dotted paths.  

This is one of the main safety properties of the design: the evaluator is intentionally not a general Python execution engine.  

## Error behavior

The class raises standard Python exceptions with explicit messages:  

| Situation | Exception |
| --- | --- |
| Unsupported constant type | `TypeError` |
| Unsupported binary operator | `TypeError` |
| Unsupported unary operator | `TypeError` |
| Unsupported AST syntax | `TypeError` |
| Missing top-level name | `KeyError` |
| Invalid dotted-path structure | `TypeError` |
| Missing path element or non-dict traversal | `KeyError` |

Because parsing happens in `__init__`, malformed expression syntax would also cause `ast.parse(...)` to raise a `SyntaxError` during object creation.  

## Security characteristics

The implementation is much safer than evaluating user formulas with `eval()`, because it parses the input and only executes whitelisted AST nodes and operators. It never allows arbitrary function calls or unrestricted attribute access.  

The effective security model is:  

- Parse once using `ast.parse(..., mode="eval")`.  
- Recursively interpret only approved AST node types.  
- Resolve values only from the supplied dictionary context.  
- Reject anything outside the supported mini-language.  

This makes the class appropriate for controlled configuration expressions in numerical workflows.  

## Typical use case

In a reactor or optimization workflow, this class can be used to define dependent configuration values based on other values in the same case context. For example, one inlet species fraction can be defined from another using a short arithmetic expression and resolved at runtime against the current context dictionary.  

Example:  

```python
ctx = {
    "inlet": {
        "specie": {
            "so2": 0.12,
            "o2": 0.18,
        }
    },
    "expr1": 0.5,
}

expr = UserExpression("1.0 - inlet.specie.so2 - inlet.specie.o2")
value = expr.value(ctx)
```

## Minimal API summary

| Item | Purpose |
| --- | --- |
| `_ALLOWED_BINOPS` | Defines the accepted binary arithmetic operators. |
| `_ALLOWED_UNARYOPS` | Defines the accepted unary arithmetic operators. |
| `UserExpression.__init__()` | Stores and parses the expression string. |
| `UserExpression.value()` | Evaluates the expression and returns `float`. |
| `UserExpression._eval_node()` | Recursive AST evaluator. |
| `UserExpression._resolve_name()` | Resolves top-level names from context. |
| `UserExpression._resolve_attribute_chain()` | Resolves dotted dictionary paths from context. |
