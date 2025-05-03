#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const axios = require('axios');
const csv = require('csv-parser');
const { createObjectCsvWriter } = require('csv-writer');

// Configuration
const CSV_FILE_PATH = path.join(__dirname, '..', 'model', 'games.csv');
const API_URL = 'https://predict-score-misho-761671483637.us-central1.run.app'; // Change this if your Flask app is running on a different port/host

/**
 * Load games from CSV file
 * @param {string} filePath - Path to the CSV file
 * @returns {Promise<Array>} - Array of games
 */
async function loadGamesFromCsv(filePath) {
  return new Promise((resolve, reject) => {
    const games = [];
    
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (row) => games.push(row))
      .on('end', () => {
        console.log(`Loaded ${games.length} games from ${filePath}`);
        resolve(games);
      })
      .on('error', (error) => {
        console.error(`Error loading CSV file: ${error.message}`);
        reject(error);
      });
  });
}

/**
 * Call the prediction API for a specific game
 * @param {string} homeTeam - Home team name
 * @param {string} awayTeam - Away team name
 * @returns {Promise<Object>} - Prediction result
 */
async function predictGame(homeTeam, awayTeam) {
  try {
    const formData = new URLSearchParams();
    formData.append('home_team', homeTeam);
    formData.append('away_team', awayTeam);
    
    const response = await axios.post(API_URL, formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Requested-With': 'XMLHttpRequest'
      }
    });
    
    if (response.status === 200) {
      const result = response.data;
      if (result.error) {
        return { error: result.error };
      } else {
        return result.prediction;
      }
    } else {
      return { error: `API returned status code ${response.status}` };
    }
  } catch (error) {
    return { error: `API request failed: ${error.message}` };
  }
}

/**
 * Format the prediction result for display
 * @param {string} date - Game date
 * @param {string} awayTeam - Away team name
 * @param {string} homeTeam - Home team name
 * @param {Object} prediction - Prediction result
 * @returns {string} - Formatted prediction
 */
function formatGamePrediction(date, awayTeam, homeTeam, prediction) {
  if (prediction.error) {
    return `${date} | ${awayTeam} @ ${homeTeam} | ERROR: ${prediction.error}`;
  }
  
  try {
    const awayMean = prediction.PredictedMeans.away_mean;
    const homeMean = prediction.PredictedMeans.home_mean;
    const awayWinProb = prediction.WinProbabilities.away_win_probability;
    const homeWinProb = prediction.WinProbabilities.home_win_probability;
    const totalLine = prediction.OverUnder.total_line;
    
    let result = `${date} | ${awayTeam} (${awayMean}) @ ${homeTeam} (${homeMean}) | `;
    result += `Win Prob: ${awayTeam}=${awayWinProb}%, ${homeTeam}=${homeWinProb}% | `;
    result += `Total: ${totalLine} (O/U: ${prediction.OverUnder.over_probability_percent}%/${prediction.OverUnder.under_probability_percent}%)`;
    
    return result;
  } catch (error) {
    return `${date} | ${awayTeam} @ ${homeTeam} | ERROR formatting prediction: ${error.message}`;
  }
}

/**
 * Main function to process all games
 */
async function main() {
  console.log('College Baseball Prediction Script');
  console.log('==================================');
  
  try {
    // Load games from CSV
    const games = await loadGamesFromCsv(CSV_FILE_PATH);
    
    // Create results directory if it doesn't exist
    const resultsDir = path.join(__dirname, 'results');
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true });
    }
    
    // File to save results
    const timestamp = new Date().toISOString().replace(/[:.]/g, '').split('T').join('_').slice(0, 15);
    const resultsFile = path.join(resultsDir, `predictions_${timestamp}.csv`);
    
    // Set up CSV writer
    const csvWriter = createObjectCsvWriter({
      path: resultsFile,
      header: [
        { id: 'date', title: 'Date' },
        { id: 'awayTeam', title: 'AwayTeam' },
        { id: 'homeTeam', title: 'HomeTeam' },
        { id: 'awayMean', title: 'AwayMean' },
        { id: 'homeMean', title: 'HomeMean' },
        { id: 'awayWinProb', title: 'AwayWinProb' },
        { id: 'homeWinProb', title: 'HomeWinProb' },
        { id: 'totalLine', title: 'TotalLine' },
        { id: 'overProb', title: 'OverProb' },
        { id: 'underProb', title: 'UnderProb' }
      ]
    });
    
    // Process each game
    const numGames = games.length;
    const results = [];
    
    for (let i = 0; i < numGames; i++) {
      const game = games[i];
      const date = game.Date;
      const awayTeam = game.AwayTeam;
      const homeTeam = game.HomeTeam;
      
      console.log(`\nProcessing game ${i+1}/${numGames}: ${awayTeam} @ ${homeTeam}`);
      const prediction = await predictGame(homeTeam, awayTeam);
      
      if (prediction.error) {
        console.log(`  Error: ${prediction.error}`);
        results.push({
          date,
          awayTeam,
          homeTeam,
          awayMean: 'ERROR',
          homeMean: 'ERROR',
          awayWinProb: 'ERROR',
          homeWinProb: 'ERROR',
          totalLine: 'ERROR',
          overProb: 'ERROR',
          underProb: 'ERROR'
        });
      } else {
        // Format and print the prediction
        const formattedResult = formatGamePrediction(date, awayTeam, homeTeam, prediction);
        console.log(`  ${formattedResult}`);
        
        // Save result to CSV
        results.push({
          date,
          awayTeam,
          homeTeam,
          awayMean: prediction.PredictedMeans.away_mean,
          homeMean: prediction.PredictedMeans.home_mean,
          awayWinProb: prediction.WinProbabilities.away_win_probability,
          homeWinProb: prediction.WinProbabilities.home_win_probability,
          totalLine: prediction.OverUnder.total_line,
          overProb: prediction.OverUnder.over_probability_percent,
          underProb: prediction.OverUnder.under_probability_percent
        });
      }
    }
    
    // Write results to CSV file
    await csvWriter.writeRecords(results);
    console.log(`\nPredictions complete! Results saved to ${resultsFile}`);
    
  } catch (error) {
    console.error(`Error: ${error.message}`);
  }
}

// Run the main function
main().catch(error => console.error(error)); 