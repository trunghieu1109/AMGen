async def forward_40(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose the demographic data to logically determine which gender group is larger, females or males, based on the given ratios and age distributions."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, decomposing demographic data, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Control Flow 1: start_loop
    loop_outputs = []
    for i in range(self.max_sc):
        cot_sc_instruction2 = f"Subtask 2: Iteratively evaluate and refine the initial outcome from Subtask 1 to enhance clarity, consistency, and completeness regarding which gender group is larger. Iteration {i+1}."
        cot_sc_desc = {
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2 = await self.sc_cot(
            subtask_id=f"subtask_2_{i+1}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results2['cot_agent'][0].id}, iteration {i+1}, refining outcome, thinking: {results2['list_thinking'][0]}; answer: {results2['list_answer'][0]}")
        sub_tasks.append(f"Subtask 2 iteration {i+1} output: thinking - {results2['list_thinking'][0]}; answer - {results2['list_answer'][0]}")
        logs.append(results2['subtask_desc'])
        loop_outputs.append((results2['list_thinking'][0], results2['list_answer'][0]))
    
    # Control Flow 2: end_loop
    
    # Stage 3: Validate and Transform Output
    review_instruction3 = "Subtask 3: Review the refined outputs from iterative evaluations to ensure correctness and transform the answer to conform to the specified format (which gender group is larger: females or males)."
    review_desc3 = {
        'instruction': review_instruction3,
        'input': [taskInfo] + [item[1] for item in loop_outputs],
        'temperature': 0.0,
        'context': ['user query', 'answers from iterative refinements']
    }
    results3 = await self.review(
        subtask_id="subtask_3",
        review_desc=review_desc3
    )
    agents.append(f"Review agent {results3['review_agent'].id}, reviewing iterative outputs, feedback: {results3['feedback'].content}; correct: {results3['correct'].content}")
    sub_tasks.append(f"Subtask 3 output: feedback - {results3['feedback'].content}; correct - {results3['correct'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 2: Iterative Quality Enhancement
    cot_reflect_instruction4 = "Subtask 4: Based on the review feedback and iterative outputs, refine and enhance the clarity, consistency, and completeness of the answer about which gender group is larger."
    critic_instruction4 = "Please review the refined answer and provide any limitations or corrections needed."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo] + [item[0] for item in loop_outputs] + [item[1] for item in loop_outputs] + [results3['feedback'], results3['correct']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4 = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining answer, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback: {results4['list_feedback'][i].content}; correction: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining further, thinking: {results4['list_thinking'][i+1].content}; answer: {results4['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    # Stage 1: Consolidate and select optimal output
    aggregate_instruction5 = "Subtask 5: Aggregate the refined answers from the Reflexion stage and select the most coherent and consistent final result indicating which gender group is larger."
    aggregate_desc5 = {
        'instruction': aggregate_instruction5,
        'input': [taskInfo] + results4['list_answer'],
        'temperature': 0.0,
        'context': ["user query", "refined answers from subtask 4"]
    }
    results5 = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc5
    )
    agents.append(f"Aggregate agent {results5['aggregate_agent'].id}, aggregating refined answers, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    # Control Flow 3: end_sequential
    
    # Final formatting to conform to answer format
    formatter_instruction6 = "Subtask 6: Format the final answer to be concise and directly state which gender group is larger: females or males."
    formatter_desc6 = {
        'instruction': formatter_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "final aggregated answer"],
        'format': 'short and concise, without explanation'
    }
    results6 = await self.specific_format(
        subtask_id="subtask_6",
        formatter_desc=formatter_desc6
    )
    agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, formatting final answer, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs