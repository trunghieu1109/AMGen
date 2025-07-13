async def forward_28(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Subtask 1: SC-CoT to derive general formula r(D)
    cot_sc_instruction = "Sub-task 1: Define the torus meridional cross-section with R=6 and a=3, derive the formula r(D)=sqrt(D^2 - a^2)."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_think1 = []
    possible_ans1 = []
    desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents1:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_think1.append(thinking)
        possible_ans1.append(answer)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo] + possible_think1 + possible_ans1, "Sub-task 1: Synthesize and choose the most consistent derivation of r(D).", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    agents.append(f"Final Decision Agent {final_decision1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(desc1)
    print("Step 1: ", sub_tasks[-1])
    # Subtask 2: Debate to solve for D_i and D_o
    debate_instruction = "Sub-task 2: Using b=11 and the two branch offsets u0_i=3 and u0_o=9, solve sqrt(u0^2 + D^2)=11 for both branches. " +
                         "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N2 = self.max_round
    all_think2 = [[] for _ in range(N2)]
    all_ans2 = [[] for _ in range(N2)]
    desc2 = {"subtask_id": "subtask_2", "instruction": debate_instruction, "context": ["user query", thinking1, answer1], "agent_collaboration": "Debate"}
    for r in range(N2):
        for agent in debate_agents:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking1, answer1], debate_instruction, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking1, answer1] + all_think2[r-1] + all_ans2[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_think2[r].append(thinking)
            all_ans2[r].append(answer)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + all_think2[-1] + all_ans2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    agents.append(f"Final Decision Agent {final_decision2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(desc2)
    print("Step 2: ", sub_tasks[-1])
    # Subtask 3: SC-CoT to compute r_i and r_o from D_i and D_o
    cot_sc_instruction3 = "Sub-task 3: Substitute the distances D_i and D_o from subtask 2 into r(D)=sqrt(D^2 - 3^2) to compute r_i and r_o and check ordering."  
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_think3 = []
    possible_ans3 = []
    desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["user query", thinking1, answer1, thinking2, answer2], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents3:
        thinking, answer = await agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_think3.append(thinking)
        possible_ans3.append(answer)
    final_decision3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision3([taskInfo] + possible_think3 + possible_ans3, "Sub-task 3: Synthesize and choose the most consistent values for r_i and r_o.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    agents.append(f"Final Decision Agent {final_decision3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(desc3)
    print("Step 3: ", sub_tasks[-1])
    # Subtask 4: CoT to compute Δr, reduce fraction, and sum m+n
    cot_instruction4 = "Sub-task 4: Compute delta r = r_i - r_o from subtask 3, reduce to lowest terms m/n, and find m+n."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", thinking3, answer3], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs