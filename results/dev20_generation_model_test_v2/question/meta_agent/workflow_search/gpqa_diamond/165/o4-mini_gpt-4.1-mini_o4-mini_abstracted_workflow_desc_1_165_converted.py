async def forward_165(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Self-Consistency Chain-of-Thought to analyze Lagrangian and field content
    cot_sc_instruction1 = "Sub-task 1: Analyze the given Lagrangian, its field content, charge assignments, and the definition of the pseudo-Goldstone boson H2."  
    N = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "SC-CoT Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"SC-CoT agent {cot_agents1[i].id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Given all the above thinking and answers, find the most consistent analysis of the Lagrangian and field content."  
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize the consistent analysis." + final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Self-Consistency Chain-of-Thought to derive one-loop mass structure
    cot_sc_instruction2 = "Sub-task 2: Derive the general form of the one-loop effective potential contribution to the mass squared of the pseudo-Goldstone H2, showing the 1/(8π²(x²+v²)) factor and the sum of fourth powers of masses."  
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "SC-CoT Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"SC-CoT agent {cot_agents2[i].id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Given all the above thinking and answers, produce the most consistent derivation of the one-loop prefactor and structure."  
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize the loop mass structure." + final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Self-Consistency Chain-of-Thought to identify specific loop contributors
    cot_sc_instruction3 = "Sub-task 3: Identify which particles (h1, W, Z, H±, H0, A0, t, Ni) contribute to the loop sum, assign signs for bosons vs fermions, and label α coefficients."  
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "SC-CoT Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"SC-CoT agent {cot_agents3[i].id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Given all the above thinking and answers, determine the consistent list of loop contributors, their signs, and α assignments."  
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, "Sub-task 3: Synthesize the loop contributor identification." + final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Debate to compare with provided formula choices and select correct one
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."  
    debate_instruction4 = "Sub-task 4: Compare the derived mass-squared formula against the four provided choices and select the correct one." + debate_instr
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instruction4, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                inputs4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(inputs4, debate_instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."  
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Select the correct formula choice." + final_instr4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs