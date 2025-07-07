async def forward_15(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Extract and define essential variables from the problem statement, including counts of residents owning each item, exactly two items, exactly three items, and total residents, with context from taskInfo"
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting variables, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Derive the inclusion-exclusion formula for four sets, introduce unknown x representing residents owning all four items, and explain the formula components, with context from taskInfo and output of Subtask 1"
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    for idx in range(len(results2['list_thinking'])):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, deriving inclusion-exclusion formula, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction3 = "Subtask 3: Explicitly set up the inclusion-exclusion equation using extracted variables and given counts, algebraically solve for x (the number of residents owning all four items), and verify the solution step-by-step"
    critic_instruction3 = "Please review the algebraic solution and verification of x, provide any limitations or corrections needed"
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
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, solving inclusion-exclusion equation, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correction: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining solution, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_sc_instruction4 = "Subtask 4: Consolidate and cross-validate the solution for x from Subtask 3 by considering multiple reasoning paths and verifying consistency with problem constraints"
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    for idx in range(len(results4['list_thinking'])):
        agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, consolidating and validating solution, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction5 = "Subtask 5: Generate and validate Python code that correctly implements the inclusion-exclusion principle for four sets, solves for x, and verifies the solution against problem constraints"
    critic_instruction5 = "Review the code implementation for correctness, adherence to inclusion-exclusion principle, and validation of solution"
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc5 = {
        'instruction': critic_instruction5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc5,
        critic_desc=critic_desc5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, generating and validating code, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback: {results5['list_feedback'][i].content}; correction: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining code and validation, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    try:
        x_candidate = int(results4['answer'].content.strip())
    except:
        x_candidate = None

    if x_candidate is not None and isinstance(x_candidate, int) and x_candidate >= 0:
        cot_reflect_instruction6 = "Subtask 6: Format the computed integer x into the required output format, explicitly extracting the numeric answer from prior subtasks without defaulting to zero"
        critic_instruction6 = "Review the formatted output to ensure correct extraction and presentation of the final numeric answer"
        cot_reflect_desc6 = {
            'instruction': cot_reflect_instruction6,
            'input': [x_candidate, results5['thinking'], results5['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
        }
        critic_desc6 = {
            'instruction': critic_instruction6,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results6 = await self.reflexion(
            subtask_id="subtask_6",
            cot_reflect_desc=cot_reflect_desc6,
            critic_desc=critic_desc6,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, formatting final answer, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results6['list_feedback']))):
            agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][i].content}; correction: {results6['list_correct'][i].content}")
            if i + 1 < len(results6['list_thinking']) and i + 1 < len(results6['list_answer']):
                agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining final formatted answer, thinking: {results6['list_thinking'][i + 1].content}; answer: {results6['list_answer'][i + 1].content}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])

        final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
        return final_answer, logs
    else:
        cot_reflect_instruction7 = "Subtask 7: Handle invalid or negative computed x by generating an error explanation and clarifying ambiguities"
        critic_instruction7 = "Review the error explanation for clarity and correctness"
        cot_reflect_desc7 = {
            'instruction': cot_reflect_instruction7,
            'input': [taskInfo, results5['thinking'], results5['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
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
        agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, generating error explanation, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results7['list_feedback']))):
            agents.append(f"Critic agent {results7['critic_agent'].id}, feedback: {results7['list_feedback'][i].content}; correction: {results7['list_correct'][i].content}")
            if i + 1 < len(results7['list_thinking']) and i + 1 < len(results7['list_answer']):
                agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, refining error explanation, thinking: {results7['list_thinking'][i + 1].content}; answer: {results7['list_answer'][i + 1].content}")
        sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
        return final_answer, logs
