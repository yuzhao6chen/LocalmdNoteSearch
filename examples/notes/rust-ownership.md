---
tags: [rust, ownership, memory]
---
# Rust Ownership Notes

## Ownership

Ownership is Rust's core memory management model. Each value has one owner, and
the value is dropped when the owner goes out of scope.

## Borrowing

Borrowing lets code read or mutate data without taking ownership. Shared
references allow many readers, while mutable references require exclusive
access.

## Error Handling

Rust projects should prefer Result-based error handling over unchecked unwraps.
