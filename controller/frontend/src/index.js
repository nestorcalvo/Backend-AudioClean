import App from './components/App'
import { render } from 'react-dom'
import React, {component} from 'react';
import { ThemeProvider } from './contexts/theme'
import './index.css'

render(
  <ThemeProvider>
    <App />
  </ThemeProvider>,
  document.getElementById('app')
)