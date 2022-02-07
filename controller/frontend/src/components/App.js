// import React, {component} from 'react';
// import { render } from 'react-dom';
// import HomePage from './HomePage';

// const App = () =>{
//     return (
        
//         <div>
//             <HomePage></HomePage>
//         </div>
//     )
// }

// render(<App/>, document.getElementById('app'))
import { useContext } from 'react'
import { ThemeContext } from '../contexts/theme'
import React, {component} from 'react';
import Header from './Header/Header'
import Footer from './Footer/Footer'
import ScrollToTop from './ScrollToTop/ScrollToTop'
import './App.css'

const App = () => {
  const [{ themeName }] = useContext(ThemeContext)

  return (
    <div id='top' className={`${themeName} app`}>
      <Header />
{/* 
      <main>
        <About />
        <Projects />
        <Skills />
        <Contact />
      </main> */}

      <ScrollToTop />
      <Footer />
    </div>
  )
}

export default App