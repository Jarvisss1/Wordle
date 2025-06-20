import React from 'react'
import Dabbe from './Dabbe'

const Board = ({
  guesses,
  currentGuess,
  currentRow,
  wordToGuess,
  isEnter,
  setIsEnter,
  isGameOver
}) => {
  return (
    <div className="flex flex-col">
      {guesses.map((guess, index) => {
        return (
          <Dabbe
            w
            guess={guess ? guess : ""}
            currentGuess={currentRow === index ? currentGuess : guess}
            wordToGuess={wordToGuess}
            isEnter={isEnter}
            setIsEnter={setIsEnter}
            key={index}
            isGameOver={isGameOver}
          />
        );
      })}
    </div>
  );
};

export default Board
