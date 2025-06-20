import React from "react";

const Header = ({ headerMsg }) => {
  return (
    <div className="flex justify-center items-center bg-gray-800 text-white p-4">
      {headerMsg ? (
        <h1 className="text-xl font-bold text-green-400">{headerMsg}</h1>
      ) : (
        <h1 className="text-xl">
          <span className="text-yellow-600 font-semibold">Wordle</span>
          <span>-Play with your mates</span>
        </h1>
      )}
    </div>
  );
};

export default Header;
