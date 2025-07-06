async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_expansions = []
    cot_instruction1 = (
        "Sub-task 1: Systematically research and generate multiple candidate expansions for the acronym 'VIVA' after the 2004 name change of VIVA Media AG. "
        "Explicitly verify if 'VIVA' is an acronym or a brand name. Generate diverse candidate expansions with evidence or references, "
        "and report if no acronym expansion exists."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    n_candidates = self.max_sc
    for i in range(n_candidates):
        results1 = await self.sc_cot(
            subtask_id=f"subtask_1_{i+1}",
            cot_sc_desc=cot_agent_desc1,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results1['cot_agent'][0].id}, generating candidate expansion iteration {i+1}, thinking: {results1['list_thinking'][0]}; answer: {results1['list_answer'][0]}")
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {results1['list_thinking'][0]}; answer - {results1['list_answer'][0]}")
        logs.append(results1['subtask_desc'])
        candidate_expansions.append(results1['list_answer'][0])
    debate_instruction2 = (
        "Sub-task 2: Conduct a debate among agents to argue for and against each candidate expansion of 'VIVA' generated in Sub-task 1. "
        "Critically evaluate the plausibility and provide external evidence or references to support or refute each candidate expansion. "
        "Consider the possibility that 'VIVA' is not an acronym but a brand name."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo] + candidate_expansions,
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "candidate expansions from subtask 1"]
    }
    results2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        final_decision_desc={
            'instruction': "Sub-task 2: Make final decision on the most plausible candidate expansions after debate.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round_idx in range(self.max_round):
        for idx, agent in enumerate(results2['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round_idx}, debating candidate expansions, thinking: {results2['list_thinking'][round_idx][idx].content}; answer: {results2['list_answer'][round_idx][idx].content}")
    agents.append(f"Final Decision agent, deciding best candidate expansions, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    cot_reflect_instruction3 = (
        "Sub-task 3: Reflect on the debated candidate expansions and fact-check their correctness and validity against reliable sources or knowledge bases. "
        "Filter out incorrect or unsupported expansions and confirm if 'VIVA' is an acronym or a brand name."
    )
    critic_instruction3 = "Please review the candidate expansions and provide feedback on their factual correctness and limitations."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
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
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, fact-checking candidate expansions, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, providing feedback, thinking: {results3['list_feedback'][i].content}; answer: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining answer, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    review_instruction4 = (
        "Sub-task 4: Review the verified candidate expansions and explicitly clarify whether 'VIVA' is an acronym or a brand name. "
        "Provide a clear and complete explanation about the nature of 'VIVA' as the company's new name after rebranding."
    )
    review_desc4 = {
        'instruction': review_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc=review_desc4,
        critic_desc={
            'instruction': "Please review the clarity and completeness of the explanation about 'VIVA' being an acronym or brand name.",
            'output': ["feedback", "correct"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, reviewing clarity of final explanation, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, providing feedback, thinking: {results4['list_feedback'][i].content}; answer: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining explanation, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    cot_instruction5 = (
        "Sub-task 5: Finalize the answer by explicitly stating that 'VIVA' is the company name after the 2004 rebranding and does not stand for any acronym. "
        "Provide a concise justification referencing the rebranding context."
    )
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
    agents.append(f"CoT agent {results5['cot_agent'].id}, finalizing answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs