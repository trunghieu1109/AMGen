async def forward_165(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Extract and summarize the field content N_{iR}, S, phi, H with their representations and VEVs, and identify the pseudo-Goldstone boson H2."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1:", sub_tasks[-1])

    sc_instruction2 = "Sub-task 2: Identify all particles contributing to the one-loop Coleman-Weinberg mass of H2, listing bosons and fermions with their loop-sign conventions."
    N2 = self.max_sc
    sc_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    for agent in sc_agents2:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Synthesize and choose the most consistent identification of contributing species.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: {answer2.content}")
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": sc_instruction2, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "SC_CoT"}
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2:", sub_tasks[-1])

    sc_instruction3 = "Sub-task 3: Determine the correct overall prefactor and VEV dependence for the one-loop mass formula of H2."
    N3 = self.max_sc
    sc_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    for agent in sc_agents3:
        thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thinkings3.append(thinking3)
        possible_answers3.append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        "Sub-task 3: Synthesize and choose the most consistent prefactor and VEV dependence.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: {answer3.content}")
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": sc_instruction3, "context": ["user query", "thinking1", "answer1", "thinking2", "answer2"], "agent_collaboration": "SC_CoT"}
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3:", sub_tasks[-1])

    sc_instruction4 = "Sub-task 4: Assemble the full one-loop expression for M_h2^2 by summing bosonic and fermionic M^4 contributions with correct signs and the chosen prefactor."
    N4 = self.max_sc
    sc_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    for agent in sc_agents4:
        thinking4, answer4 = await agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_thinkings4.append(thinking4)
        possible_answers4.append(answer4)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + possible_thinkings4 + possible_answers4,
        "Sub-task 4: Synthesize and choose the most consistent assembled mass expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: {answer4.content}")
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": sc_instruction4, "context": ["user query", "thinking1", "answer1", "thinking2", "answer2", "thinking3", "answer3"], "agent_collaboration": "SC_CoT"}
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4:", sub_tasks[-1])

    debate_instruction5 = "Sub-task 5: Compare the assembled mass formula against the four provided choices and select the exact matching expression. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for agent in debate_agents5:
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1], debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Provide final answer. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: {answer5.content}")
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query", "thinking4", "answer4"], "agent_collaboration": "Debate"}
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs