async def forward_163(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    instr1 = "Sub-task 1: Extract and summarize numerical and conceptual inputs: periods, RV amplitudes, definitions, and assumptions."
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_think1 = []
    possible_ans1 = []
    sub1_desc = {
        "subtask_id": "subtask_1",
        "instruction": instr1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_agents1[i]([taskInfo], instr1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_think1.append(thinking1)
        possible_ans1.append(answer1)
    decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Given all the above thinking and answers, find the most consistent and correct summary."
    thinking1_final, answer1_final = await decision1([taskInfo] + possible_think1 + possible_ans1, final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    sub1_desc['response'] = {"thinking": thinking1_final, "answer": answer1_final}
    logs.append(sub1_desc)
    print("Step 1:", sub_tasks[-1])
    instr2 = "Sub-task 2: Formalize relevant astrophysical relations: Kepler's third law and RV amplitude equations for circular edge-on binaries."
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_think2 = []
    possible_ans2 = []
    sub2_desc = {
        "subtask_id": "subtask_2",
        "instruction": instr2,
        "context": ["user query","thinking of subtask 1","answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1_final, answer1_final], instr2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_think2.append(thinking2)
        possible_ans2.append(answer2)
    decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Given all the above thinking and answers, find the most consistent and correct formalization."
    thinking2_final, answer2_final = await decision2([taskInfo, thinking1_final, answer1_final] + possible_think2 + possible_ans2, final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    sub2_desc['response'] = {"thinking": thinking2_final, "answer": answer2_final}
    logs.append(sub2_desc)
    print("Step 2:", sub_tasks[-1])
    instr3 = "Sub-task 3: Derive expressions for total masses of each system by combining period–mass relation and RV‐amplitude relations, and compute numerical values."
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_think3 = []
    possible_ans3 = []
    sub3_desc = {
        "subtask_id": "subtask_3",
        "instruction": instr3,
        "context": ["user query","thinking of subtask 1","answer of subtask 1","thinking of subtask 2","answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_agents3[i]([taskInfo, thinking1_final, answer1_final, thinking2_final, answer2_final], instr3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_think3.append(thinking3)
        possible_ans3.append(answer3)
    decision3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Given all the above thinking and answers, synthesize the correct mass derivations and values."
    thinking3_final, answer3_final = await decision3([taskInfo, thinking1_final, answer1_final, thinking2_final, answer2_final] + possible_think3 + possible_ans3, final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final.content}")
    sub3_desc['response'] = {"thinking": thinking3_final, "answer": answer3_final}
    logs.append(sub3_desc)
    print("Step 3:", sub_tasks[-1])
    debate_instr = "Sub-task 4: Compute the ratio M_total1/M_total2 and select the best match among the provided choices." + "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents4 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_think4 = [[] for _ in range(self.max_round)]
    all_ans4 = [[] for _ in range(self.max_round)]
    sub4_desc = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr,
        "context": ["user query","thinking of subtask 3","answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for agent in debate_agents4:
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3_final, answer3_final], debate_instr, r, is_sub_task=True)
            else:
                thinking4, answer4 = await agent([taskInfo, thinking3_final, answer3_final] + all_think4[r-1] + all_ans4[r-1], debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_think4[r].append(thinking4)
            all_ans4[r].append(answer4)
    final4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4_final, answer4_final = await final4([taskInfo, thinking3_final, answer3_final] + all_think4[-1] + all_ans4[-1], final_instr4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final.content}")
    sub4_desc['response'] = {"thinking": thinking4_final, "answer": answer4_final}
    logs.append(sub4_desc)
    print("Step 4:", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4_final, answer4_final, sub_tasks, agents)
    return final_answer, logs