async def forward_157(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage0 = "Sub-task 1: Analyze and classify the molecular components and mutations described: the transcription factor domains, phosphorylation activation, mutation X (recessive loss-of-function in transactivation domain), and mutation Y (heterozygous dominant-negative in dimerization domain). Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage0 = self.max_round
    all_thinking_stage0 = [[] for _ in range(N_max_stage0)]
    all_answer_stage0 = [[] for _ in range(N_max_stage0)]
    subtask_desc0 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_stage0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage0):
        for i, agent in enumerate(debate_agents_stage0):
            if r == 0:
                thinking0, answer0 = await agent([taskInfo], debate_instr_stage0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_stage0[r-1] + all_answer_stage0[r-1]
                thinking0, answer0 = await agent(input_infos_0, debate_instr_stage0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing molecular components and mutations, thinking: {thinking0.content}; answer: {answer0.content}")
            all_thinking_stage0[r].append(thinking0)
            all_answer_stage0[r].append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_stage0[-1] + all_answer_stage0[-1], "Sub-task 1: Synthesize and choose the most consistent classification of molecular components and mutations." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    debate_instr_stage1 = "Sub-task 1: Assess the functional impact of mutation Y in the dimerization domain, focusing on how a dominant-negative mutation can interfere with wild-type protein function, especially regarding dimerization and nuclear translocation. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage1 = self.max_round
    all_thinking_stage1 = [[] for _ in range(N_max_stage1)]
    all_answer_stage1 = [[] for _ in range(N_max_stage1)]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_stage1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage1):
        for i, agent in enumerate(debate_agents_stage1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instr_stage1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo, thinking0, answer0] + all_thinking_stage1[r-1] + all_answer_stage1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instr_stage1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assessing functional impact of mutation Y, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_stage1[r].append(thinking1)
            all_answer_stage1[r].append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + all_thinking_stage1[-1] + all_answer_stage1[-1], "Sub-task 1: Synthesize and choose the most consistent assessment of mutation Y's functional impact." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_stage2 = "Sub-task 1: Based on the output from Sub-task 1 and Sub-task 0, derive the most plausible molecular phenotype caused by mutation Y based on its dominant-negative effect and domain location, considering options like loss of dimerization, aggregation, degradation, or conformational changes."
    N = self.max_sc
    cot_agents_stage2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_stage2,
        "context": ["user query", thinking0, answer0, thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_stage2[i]([taskInfo, thinking0, answer0, thinking1, answer1], cot_sc_instruction_stage2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2[i].id}, deriving molecular phenotype caused by mutation Y, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2)
        possible_thinkings.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking0, answer0, thinking1, answer1] + possible_thinkings + possible_answers, "Sub-task 1: Synthesize and choose the most consistent molecular phenotype caused by mutation Y." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_stage3 = "Sub-task 1: Combine the analysis of mutation effects and molecular phenotypes to select the most likely observed molecular phenotype from the provided choices in the presence of mutation Y. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage3 = self.max_round
    all_thinking_stage3 = [[] for _ in range(N_max_stage3)]
    all_answer_stage3 = [[] for _ in range(N_max_stage3)]
    subtask_desc3 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_stage3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage3):
        for i, agent in enumerate(debate_agents_stage3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_stage3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking_stage3[r-1] + all_answer_stage3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_stage3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting most likely molecular phenotype, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_stage3[r].append(thinking3)
            all_answer_stage3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking_stage3[-1] + all_answer_stage3[-1], "Sub-task 1: Final selection of the most likely molecular phenotype from the provided choices." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
