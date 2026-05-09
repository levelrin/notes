## Change Input Values

```js
// It's an utility function to update an input value and notify any event-driven listeners.
// A simple `element.value = "something"` may not work because it updates the DOM property only.
// A modern frontend framework (e.g., React, Vue, Angular) may not detect this change because they often rely on synthetic events or specific data-binding mechanisms to track changes in input fields.
// Thus, we may need to dispatch 'input' and 'change' events manually to ensure that any event listeners or data-binding mechanisms are properly notified of the change.
const changeInputValue = (element, value) => {
    element.value = value;
    element.dispatchEvent(new Event('input', { bubbles: true }));
    element.dispatchEvent(new Event('change', { bubbles: true }));
};
const usernameInput = document.getElementById("usernameInput");
const passwordInput = document.getElementById("passwordInput");
const loginButton = document.getElementById("loginButton");
changeInputValue(usernameInput, "username");
changeInputValue(passwordInput, "password");
loginButton.click();
```
