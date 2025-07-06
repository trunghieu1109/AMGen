async def forward_8(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Start sequential flow
    
    # Start loop flow for iterative quality enhancement
    for i in range(3):
        cot_reflect_instruction1 = f"Subtask 1: Iteratively evaluate and improve the clarity, consistency, and completeness of the initial understanding of the census data and question requirements. Iteration {i+1}."
        critic_instruction1 = "Please review the clarity, consistency, and completeness of the current understanding and provide feedback."
        cot_reflect_desc1 = {
            'instruction': cot_reflect_instruction1,
            'input': [taskInfo],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"]
        }
        critic_desc1 = {
            'instruction': critic_instruction1,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results1 = await self.reflexion(
            subtask_id=f"subtask_{i+1}",
            cot_reflect_desc=cot_reflect_desc1,
            critic_desc=critic_desc1,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, iteration {i+1}, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results1['list_feedback']))):
            agents.append(f"Critic agent {results1['critic_agent'].id}, iteration {i+1}, feedback: {results1['list_feedback'][k].content}; correct: {results1['list_correct'][k].content}")
            if k + 1 < len(results1['list_thinking']) and k + 1 < len(results1['list_answer']):
                agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, iteration {i+1}, refining final answer, thinking: {results1['list_thinking'][k + 1].content}; answer: {results1['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task {i+1} output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
        logs.append(results1['subtask_desc'])
    
    # End loop flow
    
    # Stage 0: Construct Logical Reasoning Sequence using CoT and AnswerGenerate
    cot_instruction2 = "Sub-task 4: Decompose the census data and question into a logical sequence of reasoning steps to calculate the percentage of people who were not Native American in 2000."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of iterative refinement", "answer of iterative refinement"]
    }
    results4 = await self.answer_generate(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, decomposing census data and question, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    # Stage 3: Validate Output using Review and Reflexion
    review_instruction3 = "Sub-task 5: Evaluate the calculated percentage for correctness and reliability against the census data and question context."
    review_desc3 = {
        'instruction': review_instruction3,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc3
    )
    agents.append(f"Review agent {results5['review_agent'].id}, reviewing calculation correctness, feedback: {results5['feedback'].content}; correct: {results5['correct'].content}")
    sub_tasks.append(f"Sub-task 5 output: feedback - {results5['feedback'].content}; correct - {results5['correct'].content}")
    logs.append(results5['subtask_desc'])
    
    cot_reflect_instruction5 = "Sub-task 6: Based on the review feedback, refine and validate the final percentage of people who were not Native American in 2000."
    critic_instruction5 = "Please review the refined final answer for correctness and completeness."
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['feedback'], results5['correct']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "feedback of subtask 5", "correct of subtask 5"]
    }
    critic_desc5 = {
        'instruction': critic_instruction5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc5,
        critic_desc=critic_desc5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining final answer, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][k].content}; correct: {results6['list_correct'][k].content}")
        if k + 1 < len(results6['list_thinking']) and k + 1 < len(results6['list_answer']):
            agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining final answer, thinking: {results6['list_thinking'][k + 1].content}; answer: {results6['list_answer'][k + 1].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    # Stage 2: Aggregate and select optimal output
    aggregate_instruction7 = "Sub-task 7: Aggregate the refined solutions and select the most coherent and consistent final percentage of people who were not Native American in 2000."
    aggregate_desc7 = {
        'instruction': aggregate_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "refined solutions from subtask 6"]
    }
    results7 = await self.aggregate(
        subtask_id="subtask_7",
        aggregate_desc=aggregate_desc7
    )
    agents.append(f"Aggregate agent {results7['aggregate_agent'].id}, aggregating refined solutions, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    
    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs