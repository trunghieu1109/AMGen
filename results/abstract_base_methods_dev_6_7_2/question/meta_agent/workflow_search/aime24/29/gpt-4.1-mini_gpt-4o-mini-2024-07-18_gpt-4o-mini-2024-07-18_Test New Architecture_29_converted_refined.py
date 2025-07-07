async def forward_29(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = "Subtask 1: Analyze the constraints for placing chips on the 5x5 grid, explicitly verifying that each row and each column must be uniformly colored, and that the maximality condition forbids adding any more chips without violating these constraints. Identify all possible valid configurations under these conditions."
    critic_instruction1 = "Please review the analysis of constraints and maximality condition, and provide feedback on any logical errors or overlooked conditions."
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
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, analyzing constraints and maximality, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correction: {results1['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Derive a detailed combinatorial formula for the number of valid chip placements on the 5x5 grid, based on the constraints and maximality condition analyzed in Subtask 1. Explicitly enumerate cases and verify consistency with the constraints."
    cot_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, deriving combinatorial formula, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction3 = "Subtask 3: Critically evaluate and cross-validate the counting formula and result from Subtask 2, checking for logical consistency, correctness, and alignment with the constraints and maximality condition."
    critic_instruction3 = "Please review the counting formula and result, provide feedback on any errors or inconsistencies, and suggest corrections if needed."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc3 = {
        'instruction': critic_instruction3,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, evaluating counting result, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correction: {results3['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    results4_list = []
    for i in range(self.max_sc):
        cot_instruction4 = "Subtask 4: Provide explicit examples or a systematic method to generate valid chip configurations on the 5x5 grid that satisfy the constraints and maximality condition, illustrating the reasoning from previous subtasks."
        cot_desc4 = {
            'instruction': cot_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results4 = await self.cot(
            subtask_id="subtask_4",
            cot_agent_desc=cot_desc4
        )
        agents.append(f"CoT agent {results4['cot_agent'].id}, generating examples, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        results4_list.append(results4)
        sub_tasks.append(f"Subtask 4 output iteration {i+1}: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])

    cot_sc_instruction5 = "Subtask 5: Aggregate candidate outputs from Subtask 4, weigh reasoning quality, detect consensus or conflicts, and produce a consistent and reliable summary of valid configurations and counts."
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo] + [r['answer'] for r in results4_list],
        'temperature': 0.5,
        'context': ["user query", "candidate outputs from subtask 4"]
    }
    results5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_sc_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results5['list_thinking']):
        agents.append(f"CoT-SC agent {results5['cot_agent'][idx].id}, aggregating examples, thinking: {results5['list_thinking'][idx]}; answer: {results5['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_instruction6 = "Subtask 6: Implement a Python program that counts the number of valid chip placements on the 5x5 grid, enforcing the constraints that each row and column is uniformly colored and the maximality condition is satisfied. Provide clear reasoning steps in code comments."
    cot_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"],
        'entry_point': "count_valid_configurations"
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, generating counting code, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; code - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_reflect_instruction7 = "Subtask 7: Review and refine the Python code and reasoning from Subtask 6, ensuring correctness, completeness, and alignment with the problem constraints and maximality condition."
    critic_instruction7 = "Please provide feedback on the code correctness, suggest improvements, and confirm the final counting logic."
    cot_reflect_desc7 = {
        'instruction': cot_reflect_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    critic_desc7 = {
        'instruction': critic_instruction7,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results7 = await self.reflexion(
        subtask_id="subtask_7",
        cot_reflect_desc=cot_reflect_desc7,
        critic_desc=critic_desc7,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, reviewing code, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results7['list_feedback']))):
        agents.append(f"Critic agent {results7['critic_agent'].id}, feedback: {results7['list_feedback'][i].content}; correction: {results7['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_instruction8 = "Subtask 8: Extract the final numeric answer for the number of valid chip placements from the reviewed results, presenting it clearly and concisely without additional explanation."
    cot_desc8 = {
        'instruction': cot_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
    }
    results8 = await self.cot(
        subtask_id="subtask_8",
        cot_agent_desc=cot_desc8
    )
    agents.append(f"CoT agent {results8['cot_agent'].id}, extracting final numeric answer, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    cot_instruction9 = "Subtask 9: Format the final answer to strictly comply with output requirements, ensuring clarity and correctness."
    cot_desc9 = {
        'instruction': cot_instruction9,
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
    }
    results9 = await self.cot(
        subtask_id="subtask_9",
        cot_agent_desc=cot_desc9
    )
    agents.append(f"CoT agent {results9['cot_agent'].id}, formatting final answer, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
    sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
