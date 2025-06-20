import React, { useState, useEffect } from "react";

const Dabbe = ({ currentGuess, wordToGuess, isEnter,isGameOver }) => {
  const [correctValueIndexes, setCorrectValueIndexes] = useState([]);
  const [closeValueIndexes, setCloseValueIndexes] = useState([]);

  useEffect(() => {
    if (isEnter) {
      const newCorrectValueIndexes = [];
      const newCloseValueIndexes = [];

      for (let i = 0; i < currentGuess?.length; i++) {
        if (currentGuess[i] === wordToGuess[i]) {
          newCorrectValueIndexes.push(i);
        } else if (wordToGuess.includes(currentGuess[i])) {
          newCloseValueIndexes.push(i);
        }
      }

      setCorrectValueIndexes(newCorrectValueIndexes);
      setCloseValueIndexes(newCloseValueIndexes);
    }
  }, [isEnter, currentGuess, wordToGuess, isGameOver]);

  const guessArray = (currentGuess ?? "").padEnd(5).split("");

  return (
    <div className="flex">
      {guessArray.map((letter, index) => (
        <div
          key={index}
          className={`uppercase bg-gray-300 w-10 h-10 border-2 border-gray-500 flex items-center justify-center text-2xl font-bold m-1 rounded ${
            correctValueIndexes.includes(index)
              ? `bg-green-300`
              : closeValueIndexes.includes(index)
              ? `bg-yellow-300`
              : `bg-gray-300`
          }`}
        >
          {letter}
        </div>
      ))}
    </div>
  );
};

export default Dabbe;
