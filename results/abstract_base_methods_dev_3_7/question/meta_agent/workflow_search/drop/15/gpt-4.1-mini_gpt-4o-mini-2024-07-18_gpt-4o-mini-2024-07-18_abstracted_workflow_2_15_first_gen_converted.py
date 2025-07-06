async def forward_15(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []
    scoring_plays = []
    touchdown_pass_yardages = []
    
    cot_instruction1 = "Subtask 1: Extract all scoring plays from the passage, identifying the type of score (touchdown pass, touchdown run, field goal) and the yardage if applicable." 
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting scoring plays, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    scoring_plays_text = results1['answer'].content
    
    # Parse scoring plays from the answer content (assuming a list or structured text)
    # For safety, treat as text lines
    scoring_plays = scoring_plays_text.strip().split('\n')
    
    for idx, play in enumerate(scoring_plays, start=1):
        cot_instruction2 = f"Subtask 2: For scoring play #{idx}, determine if it is a touchdown pass and if so, extract the yardage of the pass. Scoring play text: {play}"
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [taskInfo, play],
            'temperature': 0.0,
            'context': ["user query", "scoring play extraction"]
        }
        results2 = await self.cot(
            subtask_id=f"subtask_2_{idx}",
            cot_agent_desc=cot_agent_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, analyzing scoring play #{idx}, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask 2_{idx} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        
        # Conditional check if touchdown pass
        if 'touchdown pass' in results2['answer'].content.lower():
            # Extract yardage from answer content
            cot_instruction3 = f"Subtask 3: If scoring play #{idx} is a touchdown pass, record the yardage from the answer: {results2['answer'].content}. Otherwise, skip."
            cot_agent_desc3 = {
                'instruction': cot_instruction3,
                'input': [taskInfo, results2['answer'].content],
                'temperature': 0.0,
                'context': ["user query", "touchdown pass yardage extraction"]
            }
            results3 = await self.cot(
                subtask_id=f"subtask_3_{idx}",
                cot_agent_desc=cot_agent_desc3
            )
            agents.append(f"CoT agent {results3['cot_agent'].id}, recording yardage for scoring play #{idx}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
            sub_tasks.append(f"Subtask 3_{idx} output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
            logs.append(results3['subtask_desc'])
            try:
                yardage = int(''.join(filter(str.isdigit, results3['answer'].content)))
                touchdown_pass_yardages.append(yardage)
            except:
                pass
    
    # Stage 4: Aggregate all recorded touchdown pass yardages and select the longest one
    if touchdown_pass_yardages:
        longest_yardage = max(touchdown_pass_yardages)
    else:
        longest_yardage = 0
    
    # Stage 5: Review the selected longest touchdown pass yardage for correctness and consistency
    cot_reflect_instruction5 = f"Subtask 5: Review the longest touchdown pass yardage {longest_yardage} yards for correctness and consistency with the passage." 
    critic_instruction5 = "Please review the correctness and consistency of the longest touchdown pass yardage extracted." 
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, str(longest_yardage)],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "longest touchdown pass yardage"]
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
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, reviewing longest yardage, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback round {i}, thinking: {results5['list_feedback'][i].content}; answer: {results5['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining review round {i}, thinking: {results5['list_thinking'][i+1].content}; answer: {results5['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    # Stage 6: Generate the final concise answer stating the longest touchdown pass yardage
    debate_instruction6 = f"Subtask 6: Based on the reviewed longest touchdown pass yardage, generate the final concise answer stating the longest touchdown pass yardage in yards." 
    final_decision_instruction6 = "Subtask 6: Make final decision on the longest touchdown pass yardage answer." 
    debate_desc6 = {
        'instruction': debate_instruction6,
        'context': ["user query", results5['thinking'].content, results5['answer'].content],
        'input': [taskInfo, str(longest_yardage)],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc6 = {
        'instruction': final_decision_instruction6,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        final_decision_desc=final_decision_desc6,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results6['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, generating final answer, thinking: {results6['list_thinking'][round][idx].content}; answer: {results6['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
