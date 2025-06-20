import { useEffect, useState } from "react";
import "./App.css";

import Header from "./components/Header";
import Board from "./components/Board";

function App() {
  const [wordToGuess, setWordToGuess] = useState(""); // State for the word to guess
  const [guessWords, setGuessWords] = useState(Array(5).fill(null));
  const [currentGuess, setCurrentGuess] = useState("");
  const [currentRow, setCurrentRow] = useState(0);
  const [isEnter, setIsEnter] = useState(false);
  const [isGameOver, setIsGameOver] = useState(false);
  const [headerMsg, setHeaderMsg] = useState(null);

  // Fetch random word from API
  useEffect(() => {
    const fetchWord = async () => {
      try {
        const response = await fetch(
          "https://random-word-api.vercel.app/api?words=1&length=5"
        );
        const data = await response.json();
        setWordToGuess(data[0].toUpperCase()); // Set the word in uppercase
        console.log("Word to Guess:", data[0].toUpperCase());
      } catch (error) {
        console.error("Error fetching word:", error);
      }
    };

    fetchWord();
  }, []);

  const handleReplay = () => {
    setGuessWords(Array(5).fill(null));
    setCurrentGuess("");
    setCurrentRow(0);
    setIsGameOver(false);
    setHeaderMsg(null);
    window.location.reload();
  };

  useEffect(() => {
    const handleKeydown = (e) => {
      if (
        e.key === " " ||
        e.key === "Shift" ||
        e.key === "Control" ||
        e.key === "Alt" ||
        e.key === "Meta" ||
        isGameOver
      )
        return;

      if (e.key === "Enter") {
        if (currentGuess.length === 5) {
          setIsEnter(true);
          setGuessWords((prev) => {
            const updated = [...prev];
            updated[currentRow] = currentGuess.toUpperCase();
            return updated;
          });

          if (currentGuess.toUpperCase() === wordToGuess) {
            setIsGameOver(true);
            setHeaderMsg("Congratulations! You've won!");
            console.log("Congratulations! You've guessed the word!");
          } else if (currentRow === 4) {
            setIsGameOver(true);
            setHeaderMsg(`Game Over! The word was: ${wordToGuess}`);
          }

          setCurrentGuess("");
          setCurrentRow((old) => old + 1);
        }
      } else if (e.key === "Backspace") {
        setCurrentGuess((old) => old.slice(0, -1));
      } else if (currentGuess.length < 5) {
        setCurrentGuess((prev) => (prev + e.key).toUpperCase());
      }
    };

    window.addEventListener("keydown", handleKeydown);

    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  }, [currentGuess, currentRow, wordToGuess]);

  useEffect(() => {
    if (isEnter) {
      const timer = setTimeout(() => setIsEnter(false), 100); // Reset after processing
      return () => clearTimeout(timer);
    }
  }, [isEnter]);

  return (
    <>
      <Header headerMsg={headerMsg} />
      <div className="flex flex-col items-center h-screen">
        <Board
          guesses={guessWords}
          currentGuess={currentGuess}
          isEnter={isEnter}
          currentRow={currentRow}
          wordToGuess={wordToGuess}
          isGameOver={isGameOver}
        />
        {isGameOver && (
          <button
            type="submit"
            className="bg-blue-300 rounded-sm p-2 my-5 text-black hover:bg-blue-400 hover:cursor-pointer hover:text-white"
            onClick={handleReplay}
          >
            Replay
          </button>
        )}
      </div>
    </>
  );
}

export default App;
