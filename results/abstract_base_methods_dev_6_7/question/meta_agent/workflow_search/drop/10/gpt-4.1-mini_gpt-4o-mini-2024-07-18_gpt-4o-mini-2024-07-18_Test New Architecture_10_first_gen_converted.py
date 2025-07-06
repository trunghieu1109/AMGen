async def forward_10(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Control Flow 1: start_loop
    refined_solutions = []
    for i in range(3):
        # Stage 0: Subtask 1 - Iteratively evaluate and refine extraction of all field goal yardages made by Nate Kaeding using Reflexion and Revise
        reflexion_instruction = f"Subtask 1: Iteratively evaluate and refine the extraction of all field goal yardages made by Nate Kaeding from the passage to ensure clarity, consistency, and completeness. Iteration {i+1}."
        reflexion_desc = {
            'instruction': reflexion_instruction,
            'input': [taskInfo] + refined_solutions,
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"]
        }
        critic_instruction = "Please review the refined extraction for clarity, consistency, and completeness, and provide feedback."
        critic_desc = {
            'instruction': critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_reflexion = await self.reflexion(
            subtask_id=f"subtask_1_iteration_{i+1}",
            cot_reflect_desc=reflexion_desc,
            critic_desc=critic_desc,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, refining extraction, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, iteration {i+1}, feedback: {results_reflexion['list_feedback'][k].content}; correct: {results_reflexion['list_correct'][k].content}")
            if k + 1 < len(results_reflexion['list_thinking']) and k + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, refining final answer, thinking: {results_reflexion['list_thinking'][k + 1].content}; answer: {results_reflexion['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
        refined_solutions.append(results_reflexion['answer'].content)
    
    # Control Flow 2: end_loop
    
    # Stage 1: Subtask 2 - Decompose refined info into ordered logical sequence to calculate total yards using CoT and AnswerGenerate
    cot_instruction2 = "Subtask 2: Decompose the refined information into an ordered logical sequence to calculate the total yards of field goals made by Nate Kaeding."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, refined_solutions],
        'temperature': 0.0,
        'context': ["user query", "refined extraction results"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, decomposing refined info, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    # Stage 2: Subtask 3 - Aggregate multiple variant outputs and select most coherent total yardage
    aggregate_instruction3 = "Subtask 3: Aggregate multiple variant outputs from the reasoning sequence and select the most coherent and consistent final total yardage result."
    aggregate_desc3 = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']] + refined_solutions,
        'temperature': 0.0,
        'context': ["user query", "reasoning outputs", "refined extraction results"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc3
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, aggregating outputs, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 3: Subtask 4 (optional) - Evaluate final total yardage output for correctness and reliability using Review and Reflexion
    if hasattr(self, 'review') and hasattr(self, 'reflexion'):
        review_instruction4 = "Subtask 4: Evaluate the final total yardage output against correctness and reliability criteria to validate the answer."
        review_desc4 = {
            'instruction': review_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.0,
            'context': ["user query", "final aggregated output"]
        }
        results4 = await self.review(
            subtask_id="subtask_4",
            review_desc=review_desc4
        )
        agents.append(f"Review agent {results4['review_agent'].id}, reviewing final output, feedback: {results4['feedback'].content}; correct: {results4['correct'].content}")
        sub_tasks.append(f"Sub-task 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
        logs.append(results4['subtask_desc'])
        
        reflexion_instruction4 = "Subtask 4: Based on review feedback, refine the final total yardage output to improve correctness and reliability."
        reflexion_desc4 = {
            'instruction': reflexion_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer'], results4['feedback'], results4['correct']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "final output", "review feedback"]
        }
        critic_instruction4 = "Please review the refined final total yardage output and provide any remaining limitations."
        critic_desc4 = {
            'instruction': critic_instruction4,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results4_refined = await self.reflexion(
            subtask_id="subtask_4_refine",
            cot_reflect_desc=reflexion_desc4,
            critic_desc=critic_desc4,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results4_refined['cot_agent'].id}, refining final output, thinking: {results4_refined['list_thinking'][0].content}; answer: {results4_refined['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results4_refined['list_feedback']))):
            agents.append(f"Critic agent {results4_refined['critic_agent'].id}, feedback: {results4_refined['list_feedback'][k].content}; correct: {results4_refined['list_correct'][k].content}")
            if k + 1 < len(results4_refined['list_thinking']) and k + 1 < len(results4_refined['list_answer']):
                agents.append(f"Reflexion CoT agent {results4_refined['cot_agent'].id}, refining final answer, thinking: {results4_refined['list_thinking'][k + 1].content}; answer: {results4_refined['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task 4 refine output: thinking - {results4_refined['thinking'].content}; answer - {results4_refined['answer'].content}")
        logs.append(results4_refined['subtask_desc'])
        final_thinking = results4_refined['thinking']
        final_answer = results4_refined['answer']
    else:
        final_thinking = results3['thinking']
        final_answer = results3['answer']
    
    final_result = await self.make_final_answer(final_thinking, final_answer, sub_tasks, agents)
    
    # Control Flow 3: end_sequential
    return final_result, logs
