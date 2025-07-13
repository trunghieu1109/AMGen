async def forward_168(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: SC_CoT
    cot_sc_instruction1 = "Sub-task 1: Summarize the original decay 2A→2B+2E+2V and the variant 2A→2B+2E+M, listing particle counts, masses, and known endpoint Q."
    cot_sc_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_sc_agents1:
        thinking_i, answer_i = await agent([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, summarizing decays, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr1 = "Sub-task 1: Synthesize and choose the most consistent summary of original and variant decays."
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, synth_instr1, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent1.id}, synthesizing summary, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Debate
    debate_instruction2 = (
        "Sub-task 2: Assess how replacing two V particles (massive) with a single massless M alters the number "
        "of final-state bodies and the available phase-space dimensionality. Given solutions to the problem from other agents, "
        "consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":debate_instruction2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents2:
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction2, r, is_sub_task=True)
            else:
                prev_think = all_thinking2[r-1]
                prev_ans = all_answer2[r-1]
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1] + prev_think + prev_ans, debate_instruction2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2_final, answer2_final = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], final_instr2, is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent2.id}, compiling debate, thinking: {thinking2_final.content}; answer: {answer2_final.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    subtask_desc2["response"] = {"thinking":thinking2_final,"answer":answer2_final}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: SC_CoT
    cot_sc_instruction3 = "Sub-task 3: Determine whether the outgoing E-particle spectrum remains continuous or becomes discrete based on the change in decay multiplicity and degrees of freedom."
    cot_sc_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction3,"context":["user query","response of subtask 2"],"agent_collaboration":"SC_CoT"}
    for agent in cot_sc_agents3:
        thinking3_i, answer3_i = await agent([taskInfo, thinking2_final, answer2_final], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, analyzing spectrum continuity, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr3 = "Sub-task 3: Synthesize and choose the most consistent determination of spectrum continuity."
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking2_final, answer2_final] + possible_thinkings3 + possible_answers3, synth_instr3, is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent3.id}, synthesizing spectrum continuity, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Reflexion
    reflect_inst4 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction4 = (
        "Sub-task 4: Combine energy conservation and mass differences to compute how the endpoint energy Q shifts when two V (massive) are replaced by one massless M, "
        "and articulate the adjusted spectral shape. " + reflect_inst4
    )
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs4 = [taskInfo, thinking1, answer1, thinking2_final, answer2_final, thinking3, answer3]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_reflect_instruction4,
                    "context":["user query","responses of previous subtasks"],"agent_collaboration":"Reflexion"}
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, initial compute, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        critic_inst = (
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        )
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content.strip() == "True":
            break
        cot_inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refinement, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs