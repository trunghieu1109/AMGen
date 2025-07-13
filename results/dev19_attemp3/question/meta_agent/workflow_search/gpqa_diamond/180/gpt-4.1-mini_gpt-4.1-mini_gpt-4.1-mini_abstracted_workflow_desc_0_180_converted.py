async def forward_180(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_1 = "Sub-task 1: Collect and analyze detailed neutrino energy spectra for each branch of the solar pp chain (pp-I, pp-II, pp-III), with emphasis on the continuous 8B neutrino spectrum from the pp-III branch, including its low-energy tail extending below 1 MeV. Avoid assuming zero 8B flux below 1 MeV."
    debate_instruction_1 = debate_instr_1 + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instruction_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing neutrino spectra, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_1[r].append(thinking1)
            all_answer_1[r].append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking_1[-1] + all_answer_1[-1], "Sub-task 1: Synthesize neutrino spectra analysis. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing neutrino spectra, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Based on the neutrino spectra analysis from Sub-task 1, quantify the fractional contributions of each pp chain branch to the neutrino fluxes in the two specified energy bands (700-800 keV and 800-900 keV), including the low-energy tail of 8B neutrinos."
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, quantifying flux contributions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "Sub-task 2: Synthesize fractional contributions of neutrino fluxes. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing flux contributions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = "Stage 2 Sub-task 1: Assess the impact of hypothetically stopping the pp-III branch on the neutrino fluxes in the two energy bands, using the quantified spectral contributions from Stage 1 Sub-task 2. Incorporate the neutrino travel time (~8.5 minutes) to understand when the flux changes would be observed at Earth. Explicitly evaluate how the cessation reduces flux in band 2 due to the 8B contribution, while band 1 remains mostly unaffected."
    N3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N3):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, assessing impact of stopping pp-III, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + possible_thinkings_3 + possible_answers_3, "Stage 2 Sub-task 1: Synthesize impact assessment. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing impact assessment, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4_1 = "Stage 3 Sub-task 1: Derive the approximate flux values or relative changes in flux for band 1 and band 2 neutrinos after stopping the pp-III branch, based on the impact assessment from Stage 2 Sub-task 1. Calculate the ratio Flux(band 1) / Flux(band 2) after the change, ensuring the ratio reflects the significant reduction in band 2 flux due to the pp-III cessation."
    N4_1 = self.max_sc
    cot_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4_1)]
    possible_answers_4_1 = []
    possible_thinkings_4_1 = []
    subtask_desc4_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_4_1,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4_1):
        thinking4_1, answer4_1 = await cot_agents_4_1[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4_1[i].id}, deriving flux ratio, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
        possible_answers_4_1.append(answer4_1)
        possible_thinkings_4_1.append(thinking4_1)
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_1, answer4_1 = await final_decision_agent_4_1([taskInfo, thinking3, answer3] + possible_thinkings_4_1 + possible_answers_4_1, "Stage 3 Sub-task 1: Synthesize flux ratio derivation. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing flux ratio, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc4_1['response'] = {"thinking": thinking4_1, "answer": answer4_1}
    logs.append(subtask_desc4_1)
    print("Step 4.1: ", sub_tasks[-1])

    reflect_inst_4_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_4_2 = "Stage 3 Sub-task 2: Combine and synthesize all previous results to produce a final answer for the approximate flux ratio Flux(band 1) / Flux(band 2) after stopping the pp-III branch. Compare the derived ratio to the given multiple-choice options, and provide a reasoned justification referencing the spectral data and flux changes. Use Reflexion to cross-validate and ensure no prior assumptions or errors persist." + reflect_inst_4_2
    cot_agent_4_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4_2 = self.max_round
    cot_inputs_4_2 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4_1, answer4_1]
    subtask_desc4_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_reflect_instruction_4_2,
        "context": ["user query", thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4_1, answer4_1],
        "agent_collaboration": "Reflexion"
    }
    thinking4_2, answer4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_reflect_instruction_4_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, synthesizing final answer, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    critic_inst_4_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_4_2):
        feedback, correct = await critic_agent_4_2([taskInfo, thinking4_2, answer4_2], "Please review and provide the limitations of provided solutions" + critic_inst_4_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4_2.extend([thinking4_2, answer4_2, feedback])
        thinking4_2, answer4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_reflect_instruction_4_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, refining final answer, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc4_2['response'] = {"thinking": thinking4_2, "answer": answer4_2}
    logs.append(subtask_desc4_2)
    print("Step 4.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4_2, answer4_2, sub_tasks, agents)
    return final_answer, logs
