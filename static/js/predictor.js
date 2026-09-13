// predictor.js - Handle form submission and result display

document.addEventListener('DOMContentLoaded', function() {
    const predictForm = document.getElementById('predictForm');
    const resultCard = document.getElementById('resultCard');
    
    console.log('Predictor.js loaded');
    console.log('Form found:', predictForm);
    console.log('Result card found:', resultCard);
    
    // Check authentication status first
    fetch('/api/debug-auth')
        .then(response => response.json())
        .then(data => {
            if (!data.authenticated) {
                console.log('User not authenticated, showing login prompt');
                const instructions = document.querySelector('.instructions');
                if (instructions) {
                    instructions.innerHTML = `
                        <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                            <strong>⚠️ Authentication Required</strong><br>
                            You need to <a href="/login" style="color: #007bff; text-decoration: underline;">log in</a> to use the credit assessment tool.
                            <br><br>
                            <a href="/login" class="btn-predict" style="display: inline-block; text-decoration: none;">Go to Login</a>
                        </div>
                    `;
                }
                
                // Disable the form when not authenticated
                if (predictForm) {
                    predictForm.style.opacity = '0.5';
                    predictForm.style.pointerEvents = 'none';
                    const submitBtn = predictForm.querySelector('.btn-predict');
                    if (submitBtn) {
                        submitBtn.disabled = true;
                        submitBtn.textContent = 'Login Required';
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error checking auth status:', error);
        });
    
    if (predictForm) {
        predictForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            console.log('Form submitted');
            
            // Validate all inputs before submission
            const inputs = predictForm.querySelectorAll('input[type="number"]');
            let isValid = true;
            
            inputs.forEach(input => {
                const value = parseFloat(input.value);
                const min = parseFloat(input.min);
                const max = parseFloat(input.max);
                
                if (isNaN(value) || value < min || value > max) {
                    input.style.borderColor = '#ff4d4d';
                    input.style.boxShadow = '0 0 0 3px rgba(255, 77, 77, 0.1)';
                    isValid = false;
                } else {
                    input.style.borderColor = '#c3cfe2';
                    input.style.boxShadow = 'none';
                }
            });
            
            if (!isValid) {
                alert('Please ensure all values are within the specified ranges.');
                return;
            }
            
            // Show loading state
            const submitBtn = predictForm.querySelector('.btn-predict');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Assessing...';
            submitBtn.disabled = true;
            
            try {
                // Collect form data
                const formData = new FormData(predictForm);
                const body = {};
                formData.forEach((value, key) => {
                    body[key] = parseFloat(value) || 0;
                });
                
                console.log('Sending data:', body);
                
                // Send prediction request
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { 
                        'Content-Type': 'application/json' 
                    },
                    body: JSON.stringify(body)
                });
                
                console.log('Response status:', response.status);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    console.error('Server error response:', errorText);
                    
                    // Check if it's an authentication error
                    if (response.status === 401 || response.status === 403) {
                        alert('Please log in to use the prediction feature. You will be redirected to the login page.');
                        window.location.href = '/login';
                        return;
                    }
                    
                    throw new Error('Server error: ' + response.status + ' - ' + errorText);
                }
                
                const result = await response.json();
                console.log('Prediction result:', result);
                
                // Display results
                const probText = document.getElementById('probText');
                const riskBadge = document.getElementById('riskBadge');
                
                console.log('Prob text element:', probText);
                console.log('Risk badge element:', riskBadge);
                
                if (probText && riskBadge) {
                    probText.textContent = `Confidence: ${result.confidence.toFixed(1)}%`;
                    riskBadge.textContent = result.risk_level_name;
                    
                    // Set badge color based on risk level
                    let badgeClass = "badge ";
                    if (result.risk_level_name.includes("Very High")) {
                        badgeClass += "bg-danger";
                    } else if (result.risk_level_name.includes("High")) {
                        badgeClass += "bg-warning";
                    } else if (result.risk_level_name.includes("Medium")) {
                        badgeClass += "bg-info";
                    } else if (result.risk_level_name.includes("Low")) {
                        badgeClass += "bg-success";
                    } else {
                        badgeClass += "bg-primary";
                    }
                    
                    riskBadge.className = badgeClass;
                    
                    // Display risk analysis and suggestions
                    displayRiskAnalysis(result.risk_level_name, result.confidence);
                    
                    // Show result card with animation
                    if (resultCard) {
                        resultCard.style.display = 'block';
                        resultCard.style.opacity = '0';
                        resultCard.style.transform = 'translateY(20px)';
                        
                        setTimeout(() => {
                            resultCard.style.transition = 'all 0.5s ease';
                            resultCard.style.opacity = '1';
                            resultCard.style.transform = 'translateY(0)';
                        }, 100);
                    } else {
                        console.error('Result card not found!');
                    }
                } else {
                    console.error('Result display elements not found!');
                }
                
            } catch (error) {
                console.error('Prediction error:', error);
                alert('Error during assessment: ' + error.message);
            } finally {
                // Reset button state
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            }
        });
    } else {
        console.error('Predict form not found!');
    }
    
    // Enhanced form validation with range feedback
    const inputs = document.querySelectorAll('.form-group input');
    inputs.forEach(input => {
        // Store original placeholder
        const originalPlaceholder = input.placeholder;
        
        input.addEventListener('blur', function() {
            const value = parseFloat(this.value);
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            
            if (this.value === '' || isNaN(value)) {
                this.style.borderColor = '#ff4d4d';
                this.style.boxShadow = '0 0 0 3px rgba(255, 77, 77, 0.1)';
                this.placeholder = originalPlaceholder; // Restore original placeholder
            } else if (value < min || value > max) {
                this.style.borderColor = '#ff4d4d';
                this.style.boxShadow = '0 0 0 3px rgba(255, 77, 77, 0.1)';
                this.placeholder = `Invalid! Use ${min} - ${max}`;
            } else {
                this.style.borderColor = '#c3cfe2';
                this.style.boxShadow = 'none';
                this.placeholder = originalPlaceholder; // Restore original placeholder
            }
        });
        
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            
            if (this.value !== '' && !isNaN(value) && value >= min && value <= max) {
                this.style.borderColor = '#c3cfe2';
                this.style.boxShadow = 'none';
            }
        });
        
        // Clear placeholder when user starts typing
        input.addEventListener('focus', function() {
            if (this.value === '') {
                this.placeholder = ''; // Clear placeholder when focused
            }
        });
        
        // Restore placeholder when user leaves without entering value
        input.addEventListener('blur', function() {
            if (this.value === '') {
                this.placeholder = originalPlaceholder; // Restore original placeholder
            }
        });
    });
    
    // Function to display risk analysis and suggestions
    function displayRiskAnalysis(riskLevel, confidence) {
        const riskAnalysis = document.getElementById('riskAnalysis');
        const riskExplanation = document.getElementById('riskExplanation');
        const suggestionsList = document.getElementById('suggestionsList');
        const nextStepsList = document.getElementById('nextStepsList');
        
        if (!riskAnalysis || !riskExplanation || !suggestionsList || !nextStepsList) {
            console.error('Risk analysis elements not found');
            return;
        }
        
        // Risk explanation based on level and confidence
        let explanation = '';
        let suggestions = [];
        let nextSteps = [];
        
        if (riskLevel.includes('Very Low')) {
            explanation = `Excellent! Your credit profile shows very low risk with ${confidence}% confidence. This indicates strong agricultural practices, stable yields, and favorable environmental conditions.`;
            suggestions = [
                'Maintain current farming practices and crop diversity',
                'Continue monitoring soil health and fertilizer usage',
                'Consider expanding operations with favorable credit terms',
                'Document successful harvests for future reference'
            ];
            nextSteps = [
                'Apply for credit with confidence',
                'Explore higher credit limits',
                'Consider investment in modern farming equipment',
                'Build long-term relationships with financial institutions'
            ];
        } else if (riskLevel.includes('Low')) {
            explanation = `Good news! Your credit profile shows low risk with ${confidence}% confidence. Your farming practices are generally sound with room for minor improvements.`;
            suggestions = [
                'Optimize fertilizer application timing',
                'Improve water management practices',
                'Consider crop rotation strategies',
                'Monitor weather patterns more closely'
            ];
            nextSteps = [
                'Apply for credit with good terms',
                'Focus on suggested improvements',
                'Maintain consistent record-keeping',
                'Build credit history gradually'
            ];
        } else if (riskLevel.includes('Medium')) {
            explanation = `Moderate risk detected with ${confidence}% confidence. While not ideal, this level indicates manageable challenges that can be addressed with proper planning.`;
            suggestions = [
                'Implement better irrigation systems',
                'Diversify crop selection for risk mitigation',
                'Improve soil testing and nutrient management',
                'Consider crop insurance options',
                'Develop contingency plans for weather events'
            ];
            nextSteps = [
                'Apply for credit with moderate terms',
                'Implement suggested improvements within 3-6 months',
                'Consider smaller credit amounts initially',
                'Provide additional documentation if requested',
                'Seek agricultural extension services'
            ];
        } else if (riskLevel.includes('High')) {
            explanation = `High risk identified with ${confidence}% confidence. This indicates significant challenges that need immediate attention before credit approval.`;
            suggestions = [
                'Address soil quality issues immediately',
                'Implement comprehensive water management',
                'Reduce crop diversity to focus on proven crops',
                'Improve financial record-keeping',
                'Consider alternative farming methods',
                'Seek professional agricultural consulting'
            ];
            nextSteps = [
                'Focus on risk mitigation before applying',
                'Implement all suggested improvements',
                'Consider smaller, secured loans initially',
                'Build relationships with agricultural experts',
                'Document all improvements for future applications'
            ];
        } else if (riskLevel.includes('Very High')) {
            explanation = `Very high risk detected with ${confidence}% confidence. Immediate intervention is required to address critical agricultural and financial challenges.`;
            suggestions = [
                'Conduct comprehensive soil and water analysis',
                'Implement emergency agricultural interventions',
                'Consider temporary crop changes',
                'Improve financial management practices',
                'Seek immediate professional assistance',
                'Develop long-term recovery plan'
            ];
            nextSteps = [
                'Focus on recovery before credit applications',
                'Implement emergency measures immediately',
                'Seek government agricultural assistance programs',
                'Work with agricultural extension services',
                'Consider alternative income sources temporarily'
            ];
        } else {
            explanation = `Risk level assessment shows ${confidence}% confidence. Please review your inputs and consider consulting with agricultural experts.`;
            suggestions = [
                'Verify all input values are accurate',
                'Consult with local agricultural experts',
                'Review farming practices and conditions',
                'Consider seasonal variations in your data'
            ];
            nextSteps = [
                'Double-check all form inputs',
                'Consult with agricultural professionals',
                'Gather additional data if needed',
                'Consider seasonal adjustments'
            ];
        }
        
        // Update the DOM elements
        riskExplanation.innerHTML = `<p>${explanation}</p>`;
        
        suggestionsList.innerHTML = '';
        suggestions.forEach(suggestion => {
            const li = document.createElement('li');
            li.textContent = suggestion;
            suggestionsList.appendChild(li);
        });
        
        nextStepsList.innerHTML = '';
        nextSteps.forEach(step => {
            const li = document.createElement('li');
            li.textContent = step;
            nextStepsList.appendChild(li);
        });
        
        // Show the risk analysis section
        riskAnalysis.style.display = 'block';
        
        // Add animation
        riskAnalysis.style.opacity = '0';
        riskAnalysis.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            riskAnalysis.style.transition = 'all 0.5s ease';
            riskAnalysis.style.opacity = '1';
            riskAnalysis.style.transform = 'translateY(0)';
        }, 200);
    }
});
