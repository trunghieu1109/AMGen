async def forward_22(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: SC_CoT to determine list length parity and middle entries
    sc1_instr = "Sub-task 1: Determine that the list length n must be even and characterize the two middle entries that average to the median m not present in the list."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    sub1 = {"subtask_id": "subtask_1", "instruction": sc1_instr, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking_i, answer_i = await cot_agents1[i]([taskInfo], sc1_instr, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_decider1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decider1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent description of n and middle entries.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    sub1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(sub1)
    agents.append(f"Final Decision Agent {final_decider1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: SC_CoT to determine required frequency f_9 of mode 9
    sc2_instr = "Sub-task 2: Determine the required frequency f_9 of the mode 9 ensuring it exceeds the frequency of any other value and satisfies the sum constraint."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    sub2 = {"subtask_id": "subtask_2", "instruction": sc2_instr, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking_j, answer_j = await cot_agents2[i]([taskInfo, thinking1, answer1], sc2_instr, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking_j.content}; answer: {answer_j.content}")
        possible_thinkings2.append(thinking_j)
        possible_answers2.append(answer_j)
    final_decider2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decider2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent frequency f_9.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    sub2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(sub2)
    agents.append(f"Final Decision Agent {final_decider2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Debate to enumerate all valid multisets
    debate3_base = "Sub-task 3: Enumerate all multisets of positive integers that include f_9 copies of 9, satisfy total sum 30, and yield the identified median m via the two middle entries."
    debate3_instr = debate3_base + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    sub3 = {"subtask_id": "subtask_3", "instruction": debate3_instr, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking_k, answer_k = await agent([taskInfo, thinking2, answer2], debate3_instr, r, is_sub_task=True)
            else:
                inputs_k = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking_k, answer_k = await agent(inputs_k, debate3_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_k.content}; answer: {answer_k.content}")
            all_thinking3[r].append(thinking_k)
            all_answer3[r].append(answer_k)
    final_decider3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decider3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    sub3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(sub3)
    agents.append(f"Final Decision Agent {final_decider3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: CoT to compute sum of squares
    cot4_instr = "Sub-task 4: Compute the sum of the squares of the elements in the unique valid multiset found in Stage 1."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    sub4 = {"subtask_id": "subtask_4", "instruction": cot4_instr, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot4_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    sub4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(sub4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs