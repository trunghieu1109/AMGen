async def forward_13(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction1 = "Subtask 1: Provide a detailed geometric analysis of the configuration of eight tangent circles of radius 34 and 2024 tangent circles of radius 1 inside triangle ABC, explicitly deriving the relationships between the circles, their arrangement, and the inradius of triangle ABC, including all relevant formulas and constraints."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query", "geometric configuration", "circle arrangement", "inradius derivation"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, detailed geometric analysis, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    candidate_thinkings = []
    candidate_answers = []
    for i in range(self.max_sc):
        cot_sc_instruction2 = f"Subtask 2.{i+1}: Generate a candidate inradius value of triangle ABC based on the geometric analysis from Subtask 1, providing detailed step-by-step derivation, assumptions, and verification of consistency with the problem's constraints."
        cot_sc_desc2 = {
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2 = await self.sc_cot(
            subtask_id=f"subtask_2_{i+1}",
            cot_sc_desc=cot_sc_desc2,
            n_repeat=self.max_sc
        )
        for idx in range(self.max_sc):
            agents.append(f"SC-CoT agent {results2['cot_agent'][idx].id}, candidate {i+1} reasoning path {idx+1}, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
            sub_tasks.append(f"Subtask 2.{i+1}.{idx+1} output: thinking - {results2['list_thinking'][idx]}; answer - {results2['list_answer'][idx]}")
        logs.append(results2['subtask_desc'])
        candidate_thinkings.extend(results2['list_thinking'])
        candidate_answers.extend(results2['list_answer'])
    
    cot_reflect_instruction3 = "Subtask 3: Reflect on all candidate inradius values generated in Subtask 2, verify their geometric consistency, check for duplicates or contradictions, and select the most plausible and justified inradius value that appears among the candidates. Provide detailed justification for the selection."
    critic_instruction3 = "Please review the selection of the inradius value from the candidates, identify any limitations or inconsistencies, and suggest corrections if necessary."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo] + candidate_thinkings + candidate_answers,
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"] + candidate_thinkings + candidate_answers
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
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, selecting valid inradius, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correction: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining answer, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    review_instruction4 = "Subtask 4: Independently review and validate the selected inradius value from Subtask 3 for correctness, completeness, and consistency with the problem constraints and geometric principles."
    review_desc4 = {
        'instruction': review_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "selected inradius from subtask 3"]
    }
    results4 = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc={
            'instruction': review_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "selected inradius from subtask 3"]
        },
        critic_desc={
            'instruction': "Please review the validation and provide feedback and correctness.",
            'output': ["feedback", "correct"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, reviewing selected inradius, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback: {results4['list_feedback'][i].content}; correct: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining validation, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    condition = results4['answer'].content.strip().lower() == 'correct'
    if condition:
        cot_reflect_instruction5 = "Subtask 5: Fill any gaps or missing elements in the solution to ensure completeness and clarity, based on the validated inradius value."
        cot_reflect_desc5 = {
            'instruction': cot_reflect_instruction5,
            'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "validated inradius", "review feedback"]
        }
        results5 = await self.reflexion(
            subtask_id="subtask_5",
            cot_reflect_desc=cot_reflect_desc5,
            critic_desc={
                'instruction': "Please review the completeness and clarity of the solution.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, filling gaps, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results5['list_feedback']))):
            agents.append(f"Critic agent {results5['critic_agent'].id}, feedback: {results5['list_feedback'][i].content}; correct: {results5['list_correct'][i].content}")
            if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
                agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
        sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
        logs.append(results5['subtask_desc'])
        final_thinking = results5['thinking']
        final_answer = results5['answer']
    else:
        cot_reflect_instruction6 = "Subtask 6: Refine and clarify the solution based on review feedback to improve accuracy and completeness."
        cot_reflect_desc6 = {
            'instruction': cot_reflect_instruction6,
            'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "aggregated solution", "review feedback"]
        }
        results6 = await self.reflexion(
            subtask_id="subtask_6",
            cot_reflect_desc=cot_reflect_desc6,
            critic_desc={
                'instruction': "Please review the refinement and provide feedback.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining solution after review, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results6['list_feedback']))):
            agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][i].content}; correct: {results6['list_correct'][i].content}")
            if i + 1 < len(results6['list_thinking']) and i + 1 < len(results6['list_answer']):
                agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining final answer, thinking: {results6['list_thinking'][i + 1].content}; answer: {results6['list_answer'][i + 1].content}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])
        final_thinking = results6['thinking']
        final_answer = results6['answer']
    
    cot_reflect_instruction7 = "Subtask 7: Format the final inradius answer as a reduced fraction m/n and compute m+n as the final numeric answer. Verify that the formatted answer matches the validated solution and adheres to the problem's requirements."
    critic_instruction7 = "Please review the formatted final answer for correctness and consistency with the validated solution."
    cot_reflect_desc7 = {
        'instruction': cot_reflect_instruction7,
        'input': [taskInfo, final_thinking.content, final_answer.content],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "final thinking", "final answer"]
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
    agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, formatting final numeric answer, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results7['list_feedback']))):
        agents.append(f"Critic agent {results7['critic_agent'].id}, feedback: {results7['list_feedback'][i].content}; correct: {results7['list_correct'][i].content}")
        if i + 1 < len(results7['list_thinking']) and i + 1 < len(results7['list_answer']):
            agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, refining formatted answer, thinking: {results7['list_thinking'][i + 1].content}; answer: {results7['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    
    final_answer_processed = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer_processed, logs
