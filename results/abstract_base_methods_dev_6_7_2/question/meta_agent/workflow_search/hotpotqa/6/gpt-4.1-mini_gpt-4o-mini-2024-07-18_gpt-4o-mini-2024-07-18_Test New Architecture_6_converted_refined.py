async def forward_6(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Subtask 1: Identify the name of Jaclyn Stapp's spouse based on the user query, and validate that the spouse is indeed the former frontman of a band. Confirm spouse's identity with multiple sources if possible."
    cot_agent_desc_1 = {
        'instruction': cot_instruction_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc_1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying Jaclyn Stapp's spouse, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_instruction_2 = "Subtask 2: Determine the band for which Jaclyn Stapp's spouse was the former frontman, verifying the band name with multiple sources or cross-checking with spouse's known associations."
    cot_agent_desc_2 = {
        'instruction': cot_instruction_2,
        'input': [taskInfo, results1['answer'].content],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc_2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, determining band of spouse {results1['answer'].content}, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    candidate_years = []
    cot_instruction_3 = "Subtask 3: Generate multiple candidate years for when the band disbanded, explicitly distinguishing between hiatus, breakup, and reunion dates. Provide confidence levels or source citations for each candidate year, and note any ambiguity or conflicting information."
    for i in range(self.max_sc):
        cot_agent_desc_3 = {
            'instruction': cot_instruction_3,
            'input': [taskInfo, results2['answer'].content],
            'temperature': 0.7,
            'context': ["user query", "band name from subtask 2"]
        }
        results3 = await self.answer_generate(
            subtask_id=f"subtask_3_{i+1}",
            cot_agent_desc=cot_agent_desc_3
        )
        agents.append(f"AnswerGenerate agent {results3['cot_agent'].id}, generating candidate disbandment year, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Subtask 3_{i+1} output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        candidate_years.append(results3['answer'].content)
    
    cot_reflect_instruction_4 = "Subtask 4: Reflect on the multiple candidate disbandment years, evaluate their consistency, reliability of sources, and ambiguity. Detect conflicting or uncertain data, and if needed, request additional data or re-query to resolve inconsistencies before final aggregation."
    critic_instruction_4 = "Please review the candidate years for inconsistencies, source reliability, and ambiguity. Provide feedback on limitations and suggest if further data is needed."
    cot_reflect_desc_4 = {
        'instruction': cot_reflect_instruction_4,
        'input': [taskInfo, results2['answer'].content] + candidate_years,
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "candidate years from subtask 3"]
    }
    critic_desc_4 = {
        'instruction': critic_instruction_4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4 = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc=cot_reflect_desc_4,
        critic_desc=critic_desc_4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, reflecting on candidate years, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, providing feedback, thinking: {results4['list_feedback'][k].content}; answer: {results4['list_correct'][k].content}")
        if k + 1 < len(results4['list_thinking']) and k + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining final aggregation, thinking: {results4['list_thinking'][k + 1].content}; answer: {results4['list_answer'][k + 1].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    cot_reflect_instruction_5 = "Subtask 5: Review and validate the aggregated disbandment year, reconcile conflicting data, and provide detailed justification with references or reasoning for any changes to the answer. Verify the final proposed year against authoritative sources."
    critic_instruction_5 = "Please critically review the aggregated disbandment year, explain rationale for acceptance or changes, and provide references or evidence supporting the final decision."
    cot_reflect_desc_5 = {
        'instruction': cot_reflect_instruction_5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc_5 = {
        'instruction': critic_instruction_5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc_5,
        critic_desc=critic_desc_5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, reviewing final aggregated year, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, providing feedback, thinking: {results5['list_feedback'][k].content}; answer: {results5['list_correct'][k].content}")
        if k + 1 < len(results5['list_thinking']) and k + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final review, thinking: {results5['list_thinking'][k + 1].content}; answer: {results5['list_answer'][k + 1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    cot_instruction_6 = "Subtask 6: Generate Python code to programmatically verify and validate the final disbandment year, including comments explaining the reasoning behind the chosen output and optionally performing data validation or source checks."
    cot_agent_desc_6 = {
        'instruction': cot_instruction_6,
        'input': [taskInfo, results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "final reviewed disbandment year"]
    }
    results6 = await self.code_generate(
        subtask_id="subtask_6",
        code_generate_desc=cot_agent_desc_6
    )
    agents.append(f"Code Generate Agent {results6['code_generate_agent'].id}, generating validation code, thinking: {results6['thinking'].content}; code: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; code - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    final_answer = await self.make_final_answer(results6['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
