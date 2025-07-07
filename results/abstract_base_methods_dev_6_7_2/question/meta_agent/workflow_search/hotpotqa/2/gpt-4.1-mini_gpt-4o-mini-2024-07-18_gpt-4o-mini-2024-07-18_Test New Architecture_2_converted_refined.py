async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidates = []

    # Stage 1: Generate candidate identifications with Reflexion to verify timeline and authoritative data
    reflexion_instruction = (
        "Sub-task 1: Identify the first governor of Missouri after the Missouri Compromise (1820), "
        "explicitly verifying the timeline of governors relative to 1820 and clarifying the meaning of 'after'. "
        "Consult authoritative historical data or timelines to confirm the exact sequence of governors and their terms."
    )
    reflexion_critic_instruction = (
        "Please review the candidate governor identification for timeline accuracy and factual correctness, "
        "highlight any inconsistencies or errors, and suggest corrections if needed."
    )
    for i in range(self.max_round):
        cot_reflect_desc = {
            'instruction': reflexion_instruction,
            'input': [taskInfo],
            'output': ["thinking", "answer"],
            'temperature': 0.5,
            'context': ["user query"]
        }
        critic_desc = {
            'instruction': reflexion_critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results1 = await self.reflexion(
            subtask_id=f"subtask_1_{i+1}",
            cot_reflect_desc=cot_reflect_desc,
            critic_desc=critic_desc,
            n_repeat=1
        )
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, iteration {i+1}, identifying first governor after Missouri Compromise, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
        agents.append(f"Critic agent {results1['critic_agent'].id}, iteration {i+1}, reviewing candidate, feedback: {results1['list_feedback'][0].content}; correction: {results1['list_correct'][0].content}")
        sub_tasks.append(f"Sub-task 1.{i+1} output: thinking - {results1['list_thinking'][0].content}; answer - {results1['list_answer'][0].content}; feedback - {results1['list_feedback'][0].content}; correction - {results1['list_correct'][0].content}")
        logs.append(results1['subtask_desc'])
        candidates.append(results1['list_answer'][0].content)

    # Stage 2: Aggregate candidates with Self-Consistency CoT and cross-validate with authoritative historical data
    sc_cot_instruction = (
        "Sub-task 2: Aggregate candidate identifications of the first governor after the Missouri Compromise, "
        "cross-validating each candidate against authoritative historical sources to ensure timeline and factual accuracy. "
        "Flag any inconsistencies and weigh answers by evidence quality rather than frequency alone."
    )
    cot_sc_desc = {
        'instruction': sc_cot_instruction,
        'input': [taskInfo] + candidates,
        'temperature': 0.5,
        'context': ["user query", "candidate solutions from subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, aggregation attempt {idx+1}, cross-validating candidates, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    # Stage 3: Review aggregated answer with Reflexion, requiring explicit evidence or citations
    reflexion_review_instruction = (
        "Sub-task 3: Review the aggregated answer for the first governor after the Missouri Compromise, "
        "providing explicit evidence or citations supporting the correctness of the answer to improve credibility."
    )
    critic_review_instruction = (
        "Please critically evaluate the evidence and citations provided, "
        "point out any gaps or errors, and suggest improvements if necessary."
    )
    cot_reflect_desc3 = {
        'instruction': reflexion_review_instruction,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc3 = {
        'instruction': critic_review_instruction,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, reviewing aggregated answer, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, providing feedback, thinking: {results3['list_feedback'][i].content}; answer: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining final answer, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    # Stage 4: Generate final validated answer using CoT, explicitly using reviewed answer and validating consistency
    cot_instruction4 = (
        "Sub-task 4: Generate the final validated answer about the first governor after the Missouri Compromise and their place of origin, "
        "explicitly using the reviewed and corrected answer from Sub-task 3 and validating output consistency before finalizing."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "reviewed answer from subtask 3"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, generating final validated answer, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
