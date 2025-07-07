async def forward_18(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Derive the locus equation for points P on the x-axis and Q on the y-axis such that segment PQ has length 1, and explicitly confirm this locus corresponds to the family \u2114 of segments PQ." 
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, deriving locus of PQ, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction2 = "Subtask 2: Verify and critically analyze the assumptions about the coverage of segment AB by the family \u2114 of segments PQ, explicitly considering which points on AB can be covered and identifying any uncovered points, correcting any incorrect assumptions." 
    critic_instruction2 = "Please review the coverage analysis and provide feedback on any incorrect assumptions or missing cases." 
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, verifying coverage of AB by family F, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][i].content}; correction: {results2['list_correct'][i].content}")
        if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining answer, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction3 = "Subtask 3: Collect and evaluate multiple candidate points C on segment AB that might be uncovered by segments in family \u2114, using diverse reasoning paths to identify consistent and valid candidates." 
    N = self.max_sc
    cot_sc_desc = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc,
        n_repeat=N
    )
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    for idx in range(N):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, evaluating candidate points for C, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    logs.append(results3['subtask_desc'])

    cot_instruction4 = "Subtask 4: Systematically generate a parametric description of all candidate points C(t) on segment AB and analyze coverage conditions by family \u2114 for each t in (0,1)." 
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, generating and analyzing parametric points C(t), thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction5 = "Subtask 5: Verify the uniqueness and coverage conditions of candidate points C(t), explicitly confirming which point is uniquely uncovered by family \u2114 and justifying this choice." 
    critic_instruction5 = "Please review the uniqueness verification and coverage justification, providing feedback and corrections if necessary." 
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
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
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, verifying uniqueness of uncovered point C, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback: {results5['list_feedback'][i].content}; correction: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final verification, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_instruction6 = "Subtask 6: Clarify and validate the coordinate units and expressions of the unique candidate point C, checking compliance with problem criteria." 
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, clarifying and validating candidate C, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_instruction7 = "Subtask 7: Simplify and confirm the expression for OC^2 as a fraction p/q with p and q relatively prime positive integers." 
    cot_agent_desc7 = {
        'instruction': cot_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7 = await self.cot(
        subtask_id="subtask_7",
        cot_agent_desc=cot_agent_desc7
    )
    agents.append(f"CoT agent {results7['cot_agent'].id}, simplifying OC^2 expression, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_instruction8 = "Subtask 8: Refine and restructure the entire solution explanation for clarity, logical coherence, and conciseness." 
    cot_agent_desc8 = {
        'instruction': cot_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
    }
    results8 = await self.cot(
        subtask_id="subtask_8",
        cot_agent_desc=cot_agent_desc8
    )
    agents.append(f"CoT agent {results8['cot_agent'].id}, refining explanation, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    answer_generate_instruction9 = "Subtask 9: Format the final numeric answer p+q according to contest output rules." 
    answer_generate_desc9 = {
        'instruction': answer_generate_instruction9,
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
    }
    results9 = await self.answer_generate(
        subtask_id="subtask_9",
        cot_agent_desc=answer_generate_desc9
    )
    agents.append(f"AnswerGenerate agent {results9['cot_agent'].id}, formatting final answer, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
    sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
