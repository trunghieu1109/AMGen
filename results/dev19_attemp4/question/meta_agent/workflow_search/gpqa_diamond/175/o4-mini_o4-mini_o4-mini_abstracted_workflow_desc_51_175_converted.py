async def forward_175(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    sc_instruction_1 = "Sub-task 1: Normalize the vector (-1,2,1), build the projector onto P's zero eigenspace, and compute P(P=0)."
    N1 = self.max_sc
    sc_agents_1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":sc_instruction_1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await sc_agents_1[i]([taskInfo], sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents_1[i].id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst_1 = "Sub-task 1: Synthesize and select the probability P(P=0)."
    thinking1, answer1 = await final_decision_1([taskInfo] + possible_thinkings1 + possible_answers1, final_inst_1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    sc_instruction_2 = "Sub-task 2: Apply the projector for P=0 to the normalized state, then renormalize the collapsed state."
    N2 = self.max_sc
    sc_agents_2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":sc_instruction_2,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await sc_agents_2[i]([taskInfo, thinking1, answer1], sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents_2[i].id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst_2 = "Sub-task 2: Synthesize and provide the renormalized collapsed state |ψ₀⟩."
    thinking2, answer2 = await final_decision_2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, final_inst_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction = "Sub-task 3: Build the projector onto Q's eigenvalue -1, compute P(Q=-1|P=0) using the collapsed state and choose the correct numeric probability. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R = self.max_round
    all_thinking3 = [[] for _ in range(R)]
    all_answer3 = [[] for _ in range(R)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_instruction,"context":["user query","thinking2","answer2"],"agent_collaboration":"Debate"}
    for r in range(R):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instruction, r, is_sub_task=True)
            else:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_decision_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst_3 = "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision_3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], final_inst_3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs