import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <p style={{fontSize: "6vmin", lineHeight: 1.8, textShadow: "black 0 10px 10px", marginTop: "-4vmin"}}>
          ABCDEFGHIJKLMNOPQRSTUVWXYZ <br></br>

          abcdefghijklmnopqrstuvwxyz <br></br>

          The five boxing wizards jump quickly
        </p>
        <div style={{height: "6vmin"}}></div>
        <p style={{fontSize: "3vmin"}}>
          Please enter a detailed description of the above font
        </p>
        <input type="text" name="description" style={{fontSize: "3vmin", minWidth: "70vmin", minHeight: "4vmin"}}></input>
      </header>
    </div>
  );
}

export default App;
