async def forward_10(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Sub-task 1: Extract and list all essential geometric elements, points, and relationships (rectangles, circle, collinearity, given lengths) from the problem statement with clear definitions and diagrammatic references if possible."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting geometric elements, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2a = "Sub-task 2a: Construct a coordinate system or diagram to precisely locate points A, B, C, D, E, F, G, H based on the given rectangles and circle, and express segment CE in terms of coordinates. Justify each step geometrically."
    cot_sc_desc2a = {
        'instruction': cot_sc_instruction2a,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2a = await self.sc_cot(
        subtask_id="subtask_2a",
        cot_sc_desc=cot_sc_desc2a,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2a['cot_agent'][idx].id}, coordinate construction, thinking: {results2a['list_thinking'][idx]}; answer: {results2a['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {results2a['thinking'].content}; answer - {results2a['answer'].content}")
    logs.append(results2a['subtask_desc'])

    cot_sc_instruction2b = "Sub-task 2b: Apply power of a point theorem, cyclic quadrilateral properties, and similar triangles to derive candidate equations relating CE to known lengths. Each candidate must include formal geometric justification."
    cot_sc_desc2b = {
        'instruction': cot_sc_instruction2b,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2b = await self.sc_cot(
        subtask_id="subtask_2b",
        cot_sc_desc=cot_sc_desc2b,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2b['cot_agent'][idx].id}, theorem application, thinking: {results2b['list_thinking'][idx]}; answer: {results2b['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {results2b['thinking'].content}; answer - {results2b['answer'].content}")
    logs.append(results2b['subtask_desc'])

    reflexion_instruction3 = "Sub-task 3: Reflect on candidate equations from Sub-tasks 2a and 2b, verify their geometric validity against problem constraints, discard inconsistent or incorrect ones, and select the most valid candidate(s) for CE. Provide detailed reasoning for acceptance or rejection."
    critic_instruction3 = "Please review the verification and filtering of candidate equations, identify any errors or inconsistencies, and provide constructive feedback."
    cot_reflect_desc3 = {
        'instruction': reflexion_instruction3,
        'input': [taskInfo, results2a['thinking'], results2a['answer'], results2b['thinking'], results2b['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b"]
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
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, verifying candidate equations, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correction: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining verification, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    reflexion_instruction4 = "Sub-task 4: Validate the selected equation(s) for CE with detailed geometric proof or algebraic verification, including auxiliary constructions or coordinate geometry confirmation."
    critic_instruction4 = "Please critically review the validation proof, identify any gaps or errors, and provide feedback for improvement."
    cot_reflect_desc4 = {
        'instruction': reflexion_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
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
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, validating solution proof, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback: {results4['list_feedback'][i].content}; correction: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining validation, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_instruction5 = "Sub-task 5: Refine the reasoning steps and explanation for the length of CE, explicitly addressing and correcting any errors identified in prior steps, ensuring clarity and logical coherence."
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc5
    )
    agents.append(f"CoT agent {results5['cot_agent'].id}, refining explanation, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_instruction6 = "Sub-task 6: Format the validated and refined solution for the length of CE into a concise, clear final answer, confirming it matches the latest validated solution."
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"],
        'format': 'short and concise, without explanation'
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, formatting final answer, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    reflexion_instruction7 = "Sub-task 7: Perform a final verification and review of the entire reasoning workflow and final answer for length CE, ensuring consistency, correctness, and adherence to problem constraints."
    critic_instruction7 = "Please provide feedback on the overall solution consistency and correctness, and suggest any final corrections if needed."
    cot_reflect_desc7 = {
        'instruction': reflexion_instruction7,
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
    agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, final verification, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results7['list_feedback']))):
        agents.append(f"Critic agent {results7['critic_agent'].id}, feedback: {results7['list_feedback'][i].content}; correction: {results7['list_correct'][i].content}")
        if i + 1 < len(results7['list_thinking']) and i + 1 < len(results7['list_answer']):
            agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, refining final verification, thinking: {results7['list_thinking'][i + 1].content}; answer: {results7['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_instruction8 = "Sub-task 8: Format the final verified answer for CE into the required presentation format, ensuring clarity, conciseness, and strict adherence to the problem's output requirements."
    cot_agent_desc8 = {
        'instruction': cot_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"],
        'format': 'short and concise, without explanation'
    }
    results8 = await self.cot(
        subtask_id="subtask_8",
        cot_agent_desc=cot_agent_desc8
    )
    agents.append(f"CoT agent {results8['cot_agent'].id}, formatting final verified answer, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
    return final_answer, logs
