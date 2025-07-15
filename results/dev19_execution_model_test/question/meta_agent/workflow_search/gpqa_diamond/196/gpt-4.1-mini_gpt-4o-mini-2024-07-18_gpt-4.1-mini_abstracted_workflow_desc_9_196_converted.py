async def forward_196(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage0 = "Sub-task 1: Extract and interpret the defining spectral features (IR and 1H NMR) of Compound X to characterize its functional groups and substitution pattern. This subtask must carefully identify all key spectral signals, including the presence of carboxylic acid functional groups and aromatic substitution patterns, without making assumptions about reaction outcomes. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage0 = self.max_round
    all_thinking_stage0 = [[] for _ in range(N_max_stage0)]
    all_answer_stage0 = [[] for _ in range(N_max_stage0)]
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
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
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing spectral features, thinking: {thinking0.content}; answer: {answer0.content}")
            all_thinking_stage0[r].append(thinking0)
            all_answer_stage0[r].append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_stage0[-1] + all_answer_stage0[-1], "Sub-task 1: Synthesize and choose the most consistent and correct spectral interpretation. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 0, analyzing spectral features, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Stage 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_stage1 = "Sub-task 1: Analyze the chemical reaction involving red phosphorus and HI with a focus on the known mechanistic pathway of decarboxylative reduction of aromatic carboxylic acids. This subtask must explicitly challenge any assumption that the carboxylic acid group remains intact and cite the mechanism by which ArCOOH is converted to ArCH3 under these conditions. The objective is to ensure a correct understanding of the reaction outcome based on established organic chemistry knowledge, avoiding the critical error of previous attempts."
    N_sc = self.max_sc
    cot_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage1 = []
    possible_thinkings_stage1 = []
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_stage1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1, answer1 = await cot_agents_stage1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_stage1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1[i].id}, analyzing reaction mechanism, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_stage1.append(answer1)
        possible_thinkings_stage1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings_stage1 + possible_answers_stage1, "Sub-task 1: Synthesize and choose the most consistent and correct reaction mechanism understanding. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 1, analyzing reaction mechanism, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Stage 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst_stage2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_stage2 = "Sub-task 1: Integrate the spectral characterization of Compound X with the mechanistic understanding of the red phosphorus and HI reaction to deduce the correct structure of the final product. This subtask must explicitly incorporate the decarboxylative reduction insight from stage_1.subtask_1 and avoid retaining the carboxylic acid group in the final structure. The integration should reconcile spectral data with the chemical transformation to produce a consistent and chemically plausible final product structure." + reflect_inst_stage2
    cot_agent_stage2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_stage2 = self.max_round
    cot_inputs_stage2 = [taskInfo, thinking0, answer0, thinking1, answer1]
    subtask_desc2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_stage2,
        "context": ["user query", thinking0, answer0, thinking1, answer1],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_stage2(cot_inputs_stage2, cot_reflect_instruction_stage2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2.id}, integrating spectral and mechanistic data, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_stage2):
        feedback, correct = await critic_agent_stage2([taskInfo, thinking2, answer2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_stage2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_stage2(cot_inputs_stage2, cot_reflect_instruction_stage2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2.id}, refining integration, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Stage 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_stage3 = "Sub-task 1: Evaluate the given candidate products against the deduced final product structure from stage_2.subtask_1 and select the best matching option. This evaluation must consider both spectral data and the correct reaction mechanism outcome, ensuring that the chosen product reflects the decarboxylative reduction and matches the integrated structural assignment. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage3 = self.max_round
    all_thinking_stage3 = [[] for _ in range(N_max_stage3)]
    all_answer_stage3 = [[] for _ in range(N_max_stage3)]
    subtask_desc3 = {
        "subtask_id": "stage_3.subtask_1",
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating candidate products, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_stage3[r].append(thinking3)
            all_answer_stage3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking_stage3[-1] + all_answer_stage3[-1], "Sub-task 1: Select the best matching candidate product. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 3, selecting final product, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Stage 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
