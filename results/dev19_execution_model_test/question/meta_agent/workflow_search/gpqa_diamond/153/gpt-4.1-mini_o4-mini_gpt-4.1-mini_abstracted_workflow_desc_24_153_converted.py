async def forward_153(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_0 = "Sub-task 1: Extract and transform the given spectral data (mass spec, IR, 1H NMR) into possible structural features and generate variant structural hypotheses consistent with these features. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0 = self.max_round
    all_thinking_0 = [[] for _ in range(N_max_0)]
    all_answer_0 = [[] for _ in range(N_max_0)]
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking_0, answer_0 = await agent([taskInfo], debate_instr_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_0[r-1] + all_answer_0[r-1]
                thinking_0, answer_0 = await agent(input_infos_0, debate_instr_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, generating hypotheses, thinking: {thinking_0.content}; answer: {answer_0.content}")
            all_thinking_0[r].append(thinking_0)
            all_answer_0[r].append(answer_0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0, answer_0 = await final_decision_agent_0([taskInfo] + all_thinking_0[-1] + all_answer_0[-1], "Sub-task 1: Synthesize and choose the most consistent structural hypotheses." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing hypotheses, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1 = "Sub-task 2: Assess the impact of the identified spectral features and isotope patterns on the proposed structural variants from Sub-task 1, refining or discarding hypotheses based on consistency with the data."
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, refining hypotheses, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo, thinking_0, answer_0] + possible_thinkings_1 + possible_answers_1, "Sub-task 2: Synthesize and choose the most consistent refined hypotheses." + " Given all the above thinking and answers, find the most consistent and correct solutions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr_2 = "Sub-task 3: Evaluate all refined structural candidates against the full set of spectral data, select the most plausible structure, and prioritize it as the reasonable structural suggestion for the unidentified drug. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_max_2)]
    all_answer_2 = [[] for _ in range(N_max_2)]
    subtask_desc_2 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking_2, answer_2 = await agent([taskInfo, thinking_0, answer_0, thinking_1, answer_1], debate_instr_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking_0, answer_0, thinking_1, answer_1] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking_2, answer_2 = await agent(input_infos_2, debate_instr_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating candidates, thinking: {thinking_2.content}; answer: {answer_2.content}")
            all_thinking_2[r].append(thinking_2)
            all_answer_2[r].append(answer_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo, thinking_0, answer_0, thinking_1, answer_1] + all_thinking_2[-1] + all_answer_2[-1], "Sub-task 3: Final selection of the most plausible structure." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final structure, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2, answer_2, sub_tasks, agents)
    return final_answer, logs
